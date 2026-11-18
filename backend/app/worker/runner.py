import logging
import shutil
import tempfile
from pathlib import Path

from app.config import get_settings
from app.core.db import SessionLocal
from app.core.errors import UserFacingError
from app.models import Job
from app.options import AI_PICKS, DEFAULTS, LAYOUTS, target_clip_count
from app.worker.queue import JobCanceled, ProgressReporter, reassigned

log = logging.getLogger(__name__)

STAGES = [
    ("download", 5), ("transcribe", 22), ("diarize", 30), ("signals", 11),
    ("sheets", 4), ("analysis", 10), ("cut", 13), ("upload", 5),
]


def stage_plan(diarize: bool = False) -> list[tuple[str, float]]:
    """[(stage, percent of the whole job)], always summing to 100. Diarization only counts when enabled."""
    stages = [(n, w) for n, w in STAGES if n != "diarize" or diarize]
    total = sum(w for _, w in stages)
    return [(n, w * 100 / total) for n, w in stages]


class StageProgress:
    """Turns 'stage X is 40% done' into one overall percentage that never moves backwards."""

    def __init__(self, reporter: ProgressReporter, diarize: bool = False):
        self.reporter = reporter
        self.ranges: dict[str, tuple[float, float]] = {}
        start = 0.0
        for name, weight in stage_plan(diarize):
            self.ranges[name] = (start, weight)
            start += weight
        self.stage = None
        self.parts, self.part, self._max = 1, 0, 0.0
        self.follow = {"bar": None}             # which progress bar currently drives the stage

    def begin(self, name: str, message: str = "", parts: int = 1) -> None:
        self.stage, self.parts, self.part, self._max = name, max(parts, 1), 0, 0.0
        self.follow["bar"] = None
        self.at(0.0, message, force=True)       # also raises JobCanceled if the job was canceled

    def next_part(self) -> None:
        self.part += 1

    def at(self, fraction: float, message: str = "", force: bool = False) -> None:
        if self.stage is None:
            return
        inner = (self.part + min(max(fraction, 0.0), 1.0)) / self.parts
        self._max = max(self._max, min(inner, 1.0))
        start, weight = self.ranges[self.stage]
        self.reporter.report(self.stage, start + weight * self._max, message, force=force)

    def end(self, message: str = "") -> None:
        self.part = self.parts
        self.at(1.0, message, force=True)


# Bars that must NOT move the job's progress (pulsing placeholders and setup steps).
_IGNORED_BARS = ("loading", "extracting audio", "decoding", "downloading model")


def make_progress_hook(sp: StageProgress):
    """Turns events from pipeline/progress.py into job progress. Follows one bar at a time."""
    state = sp.follow

    def hook(event, **data):
        try:
            if event == "log":
                return
            name, total = str(data.get("name", "")), data.get("total")

            if event == "bar_start":
                if total and not any(k in name.lower() for k in _IGNORED_BARS):
                    state["bar"] = state["bar"] or name
            elif event == "bar_update" and total and name == state["bar"]:
                sp.at(min(max(float(data.get("completed") or 0) / float(total), 0.0), 1.0))
            elif event == "bar_finish" and name == state["bar"]:
                state["bar"] = None
        except JobCanceled:
            raise                               # this is how a cancel reaches the pipeline
        except Exception:
            pass                                # a progress glitch must never fail a job

    return hook


def run_job(job_id: str, reporter: ProgressReporter) -> list[dict]:
    # Heavy imports live here so the API process and the tests never load torch / mediapipe.
    from app.core.storage import ObjectStorage
    from app.pipeline import progress
    from app.pipeline.analyze import analyze_video_for_highlights
    from app.pipeline.captions import build_ass, fonts_dir
    from app.pipeline.category import clip_hint, detect_category
    from app.pipeline.clipfit import fit_clips
    from app.pipeline.cutter import captions_supported, cut_video_clip
    from app.pipeline.diarize import assign_speakers, diarize_audio
    from app.pipeline.engagement import add_engagement, rank_windows
    from app.pipeline.faces import add_face_signals
    from app.pipeline.media import SAMPLE_RATE, load_audio, probe_video
    from app.pipeline.payload import build_payload
    from app.pipeline.reframe import track_faces
    from app.pipeline.replay import add_replay_signals
    from app.pipeline.scenes import add_scene_signals, detect_cuts
    from app.pipeline.sheets import build_contact_sheets
    from app.pipeline.signals import add_audio_signals, loudness_curves
    from app.pipeline.summary import build_local_summary, speaker_stats
    from app.pipeline.transcribe import transcribe_audio
    from app.worker.source import prepare_source

    settings = get_settings()

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            raise UserFacingError("Job not found.")
        source_type, source_key, url = job.source_type, job.source_key, job.source_url
        options = dict(job.options or {})

    # Everything the user chose comes from the job's options.
    layout = options.get("layout", DEFAULTS["layout"])
    if layout not in LAYOUTS:
        raise UserFacingError("Unsupported layout.")
    framing = options.get("framing", DEFAULTS["framing"])
    caption_style = options.get("caption_style", DEFAULTS["caption_style"])
    hook_mode = options.get("hook_mode", DEFAULTS["hook_mode"])
    hooks = hook_mode != "none"
    length = options.get("clip_length") or {}
    min_len = float(length.get("min_sec", DEFAULTS["clip_min"]))
    max_len = float(length.get("max_sec", DEFAULTS["clip_max"]))
    guidance = AI_PICKS.get(options.get("ai_picks") or "", AI_PICKS["best_overall"])
    custom_prompt = options.get("custom_prompt") or ""

    storage = ObjectStorage()
    sp = StageProgress(reporter, settings.diarize)
    progress.set_hook(make_progress_hook(sp))
    work = Path(tempfile.mkdtemp(prefix=f"reelio-{job_id}-"))
    succeeded = False

    try:
        # 1. Get the video
        sp.begin("download", "Preparing video")
        video_path, meta = prepare_source(source_type, source_key, url, sp, job_id)
        duration = probe_video(video_path).duration or meta.get("duration") or 0
        if duration > settings.max_video_seconds + 5:
            raise UserFacingError(f"Videos can be at most {settings.max_video_seconds // 60} minutes long.")
        sp.end()

        # 2. Transcription. The audio is decoded once and shared by every stage below.
        sp.begin("transcribe", "Transcribing speech")
        try:
            audio = load_audio(video_path)
        except RuntimeError as exc:
            raise UserFacingError("Couldn't read the audio of this video.") from exc
        duration = duration or len(audio) / SAMPLE_RATE

        transcript = transcribe_audio(
            audio, model_size=settings.whisper_model, language=settings.whisper_language,
            context=meta.get("context"), vocabulary=meta.get("vocabulary"),
        )
        segments = transcript.segments
        if not segments:
            raise UserFacingError("No speech was found in this video.")
        sp.end()

        # 3. Speakers (server setting, off by default)
        if settings.diarize:
            sp.begin("diarize", "Identifying speakers")
            segments = assign_speakers(segments, diarize_audio(audio))
            sp.end()

        # 4. Signals
        sp.begin("signals", "Analysing audio, faces and scenes")
        curves = loudness_curves(audio)
        add_audio_signals(segments, audio, curves)
        try:
            add_face_signals(segments, video_path)
        except JobCanceled:
            raise
        except Exception:
            log.exception("Face analysis failed; continuing without it")
            for seg in segments:
                seg["face"], seg["listener"] = None, None
        cuts = detect_cuts(video_path)
        add_scene_signals(segments, cuts)
        heatmap = meta.get("heatmap")
        add_replay_signals(segments, heatmap)
        add_engagement(segments)

        num_clips = target_clip_count(duration, min_len, max_len, settings.max_clips_per_job, options.get("num_clips"))
        windows = rank_windows(segments, min_len=min_len, max_len=max_len, top_n=max(num_clips * 2, 6))
        summary = build_local_summary(segments, cuts, curves, duration, heatmap)
        category = detect_category(meta, segments)
        sp.end()

        # 5. Contact sheets (local files only; they exist just to be shown to the model)
        sp.begin("sheets", "Building contact sheets")
        sheets = build_contact_sheets(video_path, segments, work / "sheets")
        payload = build_payload(
            video_id=job_id, duration=duration, language=transcript.language,
            segments=segments, sheets=sheets, summary=summary, windows=windows,
            speakers=speaker_stats(segments),
            extra={"title": meta.get("title", ""), "category": category, "clip_hint": clip_hint(category)},
        )
        sp.end()

        # 6. Highlights, fitted to the requested length
        sp.begin("analysis", "Finding highlights")
        try:
            found = analyze_video_for_highlights(
                payload, {s.image_id: s.path for s in sheets},
                num_clips=num_clips, min_len=min_len, max_len=max_len,
                ai_pick_guidance=guidance, custom_prompt=custom_prompt, hooks=hooks,
                model=settings.gemini_model or None,
            )
        except JobCanceled:
            raise
        except Exception:
            log.exception("AI analysis failed; using the highest-scoring windows instead")
            found = [{"clip_title": f"Highlight {n}", "hook_text": "",
                      "reasoning": "Picked from loudness, pacing and reactions (AI analysis was unavailable).",
                      "start_timestamp": w["start"], "end_timestamp": w["end"]}
                     for n, w in enumerate(windows, start=1)]

        to_cut = fit_clips(found, segments, min_len, max_len, duration)[:num_clips]
        if not to_cut:
            raise UserFacingError("No highlights could be found in this video.")
        for clip in to_cut:
            if hooks and not clip.get("hook_text"):
                clip["hook_text"] = clip["clip_title"]
        sp.end()

        # 7. Cut (captions and hook banner when asked for)
        want_captions = caption_style != "none" or hook_mode != "none"
        can_caption = want_captions and captions_supported()
        if want_captions and not can_caption:
            log.warning("This ffmpeg build has no 'ass' filter; rendering without captions or hook banners")
        fonts = fonts_dir() if can_caption else None

        sp.begin("cut", "Cutting clips", parts=len(to_cut))
        rendered = []
        for n, clip in enumerate(to_cut, start=1):
            start, end = clip["start_timestamp"], clip["end_timestamp"]

            track = track_faces(video_path, start, end) if framing == "fill" else None

            ass = None
            if can_caption:
                ass_path = work / "captions" / f"clip{n}.ass"
                if build_ass(ass_path, segments, start, end, layout, caption_style,
                             hook_text=clip.get("hook_text", "") if hook_mode == "text_banner" else ""):
                    ass = str(ass_path)

            filename = f"clip{n}_{layout.replace(':', 'x')}.mp4"
            out = cut_video_clip(video_path, start, end, str(work / "clips"), filename,
                                 layout=layout, framing=framing, track=track, subtitles=ass, fonts_dir=fonts)
            rendered.append({"n": n, "clip": clip, "filename": filename, "path": out})
            sp.next_part()
        sp.end()

        # 8. Upload
        sp.begin("upload", "Uploading clips", parts=len(rendered))
        clips = []
        for item in rendered:
            public_id = storage.upload_clip(item["path"], f"jobs/{job_id}/clips/{item['filename']}")
            c = item["clip"]
            start, end = float(c["start_timestamp"]), float(c["end_timestamp"])
            clips.append({
                "rank": item["n"],
                "title": c["clip_title"],
                "reasoning": c["reasoning"],
                "hook_text": c.get("hook_text") or None,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(end - start, 2),
                "layout": layout,
                "storage_key": public_id,
            })
            sp.next_part()
        sp.end()

        succeeded = True
        return clips

    finally:
        progress.set_hook(None)
        progress.stop()
        shutil.rmtree(work, ignore_errors=True)
        if not succeeded and not reassigned(job_id, reporter.worker_id):
            storage.delete_job_assets(job_id)       # don't leave half-finished uploads behind
