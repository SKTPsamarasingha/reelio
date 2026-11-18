"""The JSON sent to the model. Compact on purpose: values that are missing, zero or unremarkable are left out."""


def _segment_entry(i: int, seg: dict) -> dict:
    sig: dict = {}

    def put(key, value, minimum=None):
        if value is None or value is False:
            return
        if minimum is not None and value < minimum:
            return
        sig[key] = value

    put("energy", seg.get("energy"))
    put("wps", seg.get("wps"))
    put("pause_before", seg.get("pause_before"), minimum=0.8)
    put("engagement", seg.get("engagement"))
    put("reaction", seg.get("reaction_after"), minimum=0.3)
    put("cuts", seg.get("cut_count") or None)
    put("replay", seg.get("replay"))
    if seg.get("conf") is not None and seg["conf"] < 0.6:
        sig["conf"] = seg["conf"]                     # doubtful transcription

    face = seg.get("face")
    if face:
        if face["emotion"] != "neutral":
            sig["face"] = {"emotion": face["emotion"], "intensity": face["intensity"]}
            if face.get("shift"):
                sig["face"]["shift"] = True
        if face.get("count", 1) > 1:
            sig["faces_in_frame"] = face["count"]
    if seg.get("listener"):
        sig["listener_reaction"] = seg["listener"]

    entry = {"id": f"seg_{i}", "start": round(seg["start"], 2), "end": round(seg["end"], 2), "text": seg["text"]}
    if seg.get("speaker"):
        entry["speaker"] = seg["speaker"]
    entry["signals"] = sig
    return entry


def build_payload(*, video_id, duration, language, segments, sheets, summary, windows, speakers, extra=None) -> dict:
    """extra: title, category, clip_hint. Local file paths never go in here."""
    metadata = {
        "video_id": video_id,
        "duration_seconds": round(float(duration), 2),
        "language": language,
        **(extra or {}),
        "speakers": speakers,
        "local_signals_summary": summary,
        "candidate_windows": windows,
    }
    if not speakers:
        metadata.pop("speakers")

    return {
        "metadata": metadata,
        "transcript": [_segment_entry(i, seg) for i, seg in enumerate(segments)],
        "visual_context": {
            "format": "contact_sheet",
            "images": [
                {"image_id": s.image_id, "start_timestamp": s.start, "end_timestamp": s.end,
                 "grid": s.grid, "frames": s.frames}
                for s in sheets
            ],
        },
    }
