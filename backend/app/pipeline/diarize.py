"""Who speaks when (pyannote). Optional, and slow on CPU."""
import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from app.pipeline.media import SAMPLE_RATE
from app.pipeline.progress import Bar, DiarizationHook, ok

MODEL_ID = os.environ.get("DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")
_pipeline = None


def _load():
    """Loaded once per worker process."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set")

    with Bar("diarize: loading model", None):
        pipe = Pipeline.from_pretrained(MODEL_ID, token=token)
        if pipe is None:
            raise RuntimeError("Model load failed. Did you accept the terms on the Hugging Face model page?")
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
        else:
            torch.set_num_threads(os.cpu_count() or 4)
        for attr in ("segmentation_batch_size", "embedding_batch_size"):
            if hasattr(pipe, attr):
                setattr(pipe, attr, 16)
    _pipeline = pipe
    return pipe


def diarize_audio(audio: np.ndarray, num_speakers=None, min_speakers=None, max_speakers=None):
    """audio: 16 kHz mono float32. Returns [(start, end, speaker_label)]."""
    import torch

    pipe = _load()
    waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
    with DiarizationHook() as hook:
        output = pipe(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            hook=hook,
            num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers,
        )

    # Prefer the one-speaker-at-a-time result when it exists; it lines up better with a transcript.
    annotation = (
        getattr(output, "exclusive_speaker_diarization", None)
        or getattr(output, "speaker_diarization", output)
    )
    turns = [(float(t.start), float(t.end), label) for t, _, label in annotation.itertracks(yield_label=True)]
    ok(f"Diarization: {len({label for _, _, label in turns})} speaker(s), {len(turns)} turns")
    return turns


def assign_speakers(segments, turns):
    """Give each segment the speaker who talks most during it. Labels become speaker_0, speaker_1...
    in order of first appearance."""
    names: dict[str, str] = {}
    for seg in segments:
        overlap: dict[str, float] = {}
        for t0, t1, label in turns:
            amount = min(seg["end"], t1) - max(seg["start"], t0)
            if amount > 0:
                overlap[label] = overlap.get(label, 0.0) + amount

        if overlap:
            raw = max(overlap, key=overlap.get)
        else:                                           # no overlap (music): use the nearest turn
            mid = (seg["start"] + seg["end"]) / 2
            nearest = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])), default=None)
            raw = nearest[2] if nearest else None

        if raw is None:
            seg["speaker"] = "unknown"
        else:
            seg["speaker"] = names.setdefault(raw, f"speaker_{len(names)}")
    return segments
