"""YouTube's "most replayed" heatmap, mapped onto transcript segments (link sources only)."""


def add_replay_signals(segments, heatmap):
    """Adds replay = time-weighted average replay score (0-1), or None when there is no heatmap."""
    for seg in segments:
        weighted = covered = 0.0
        for bucket in heatmap or []:
            overlap = min(seg["end"], bucket["end_time"]) - max(seg["start"], bucket["start_time"])
            if overlap > 0:
                weighted += bucket["value"] * overlap
                covered += overlap
        seg["replay"] = round(weighted / covered, 2) if covered else None
    return segments
