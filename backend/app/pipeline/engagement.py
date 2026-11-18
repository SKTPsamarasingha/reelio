"""Blend the per-segment signals into one engagement score, and rank candidate clip windows."""
from app.pipeline.progress import ok
from app.pipeline.signals import scale01

# Weights sum to 1.0. Tune against videos that have a real replay heatmap.
W_PEAK, W_LEVEL, W_PACE, W_REACT, W_FACE = 0.35, 0.20, 0.15, 0.20, 0.10
MAX_INTERNAL_GAP = 6.0       # a window can't span a longer silence than this


def _expression_strength(seg) -> float:
    """Visible emotion: the main face, or (slightly less) someone reacting off-screen of the speaker."""
    face = seg.get("face") or {}
    listener = seg.get("listener") or {}
    main = float(face.get("intensity", 0.0)) if face.get("emotion") not in (None, "neutral") else 0.0
    return max(main, 0.8 * float(listener.get("intensity", 0.0)))


def add_engagement(segments):
    """Adds engagement (0-1) to each segment. Our own proxy for viewer interest, NOT viewer data."""
    raw = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        pace = min(max(((seg.get("wps") or 0.0) - 2.0) / 3.0, 0.0), 1.0) if dur >= 1.0 else 0.0
        raw.append(
            W_PEAK * (seg.get("peak") or 0.0)
            + W_LEVEL * (seg.get("energy") or 0.0)
            + W_PACE * pace
            + W_REACT * (seg.get("reaction_after") or 0.0)
            + W_FACE * _expression_strength(seg)
        )
    for seg, value in zip(segments, scale01(raw, flat=0.5)):
        seg["engagement"] = round(float(value), 2)
    ok(f"Engagement scores added to {len(segments)} segments")
    return segments


def _why(part) -> list[str]:
    """Plain-language reasons a window scored well, so the model knows what to look for."""
    tags = []
    if max((s.get("peak") or 0.0) for s in part) >= 0.6:
        tags.append("loud moment")
    if max((s.get("reaction_after") or 0.0) for s in part) >= 0.5:
        tags.append("crowd or audience reaction")
    seconds = sum(s["end"] - s["start"] for s in part) or 1.0
    if sum((s.get("wps") or 0.0) * (s["end"] - s["start"]) for s in part) / seconds >= 3.8:
        tags.append("fast delivery")
    if any(_expression_strength(s) >= 0.5 for s in part):
        tags.append("visible emotion")
    if sum(s.get("cut_count") or 0 for s in part) >= 4:
        tags.append("fast cutting")
    speakers = [s.get("speaker") for s in part if s.get("speaker") not in (None, "unknown")]
    if sum(1 for a, b in zip(speakers, speakers[1:]) if a != b) >= 3:
        tags.append("rapid back-and-forth")
    if max((s.get("replay") or 0.0) for s in part) >= 0.6:
        tags.append("viewers replayed this")
    return tags


def rank_windows(segments, min_len=15.0, max_len=60.0, top_n=8):
    """Best non-overlapping clip ranges, aligned to segment boundaries, best first."""
    cands = []
    for i in range(len(segments)):
        for j in range(i, len(segments)):
            if j > i and segments[j]["start"] - segments[j - 1]["end"] > MAX_INTERNAL_GAP:
                break
            start, end = segments[i]["start"], segments[j]["end"]
            length = end - start
            if length > max_len:
                break
            if length < min_len:
                continue

            part = segments[i: j + 1]
            weights = [max(s["end"] - s["start"], 0.01) for s in part]
            vals = [s.get("engagement") or 0.0 for s in part]
            mean = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
            score = 0.5 * mean + 0.35 * max(vals) + 0.15 * (part[-1].get("reaction_after") or 0.0)
            cands.append((score, i, j, start, end))

    cands.sort(reverse=True)
    picked = []
    for score, i, j, start, end in cands:
        if all(end <= p["start"] or start >= p["end"] for p in picked):
            picked.append({
                "start": round(float(start), 2), "end": round(float(end), 2), "score": round(float(score), 2),
                "first_segment": f"seg_{i}", "last_segment": f"seg_{j}",
                "why": _why(segments[i: j + 1]),
            })
        if len(picked) >= top_n:
            break
    return picked
