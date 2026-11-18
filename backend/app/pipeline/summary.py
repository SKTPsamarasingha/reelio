"""Whole-video highlights, pre-computed so the model doesn't have to do arithmetic across hundreds of segments."""
import numpy as np

from app.pipeline.signals import Loudness

PEAK_GAP = 6.0            # loud moments must be at least this far apart (seconds)
CUT_WINDOW = 10.0         # fast-cutting windows
EXCHANGE_WINDOW = 20.0    # back-and-forth windows


def speaker_stats(segments) -> list[dict]:
    """Talk time per speaker, biggest first. Empty when diarization was off."""
    totals: dict[str, float] = {}
    for seg in segments:
        label = seg.get("speaker")
        if label and label != "unknown":
            totals[label] = totals.get(label, 0.0) + (seg["end"] - seg["start"])
    whole = sum(totals.values()) or 1.0
    return [{"id": k, "talk_seconds": round(v, 1), "talk_share": round(v / whole, 2)}
            for k, v in sorted(totals.items(), key=lambda kv: -kv[1])]


def loudest_moments(curves: Loudness, top=6) -> list[dict]:
    """Timestamps of the strongest loudness spikes, spread out across the video."""
    if curves.empty:
        return []
    score = 0.6 * curves.spike + 0.4 * curves.level
    picked = []
    for idx in np.argsort(score)[::-1]:
        if score[idx] < 0.4:
            break
        t = float(curves.times[idx])
        if all(abs(t - p["t"]) >= PEAK_GAP for p in picked):
            picked.append({"t": round(t, 1), "strength": round(float(score[idx]), 2)})
        if len(picked) >= top:
            break
    return sorted(picked, key=lambda p: p["t"])


def _busiest_windows(times, size, step, minimum, top=3):
    """Non-overlapping windows with the most events (times sorted ascending)."""
    if not times:
        return []
    arr = np.asarray(times)
    candidates = []
    start = 0.0
    while start <= arr[-1]:
        count = int(np.sum((arr >= start) & (arr < start + size)))
        if count >= minimum:
            candidates.append((count, start))
        start += step
    candidates.sort(key=lambda c: (-c[0], c[1]))
    chosen = []
    for count, s in candidates:
        if all(s + size <= c[1] or s >= c[1] + size for c in chosen):
            chosen.append((count, s))
        if len(chosen) >= top:
            break
    return [{"start": round(s, 1), "end": round(s + size, 1), "count": c} for c, s in sorted(chosen, key=lambda c: c[1])]


def build_local_summary(segments, cuts, curves, duration, heatmap=None) -> dict:
    out: dict = {"loudest_moments": loudest_moments(curves)}

    reactions = sorted(
        ((i, s) for i, s in enumerate(segments) if (s.get("reaction_after") or 0) >= 0.5),
        key=lambda x: -x[1]["reaction_after"])[:5]
    if reactions:
        out["crowd_reactions"] = sorted(
            [{"after": f"seg_{i}", "at": round(s["end"], 1), "strength": s["reaction_after"]} for i, s in reactions],
            key=lambda r: r["at"])

    minutes = max(duration / 60.0, 1e-6)
    scene = {"total_cuts": len(cuts), "cuts_per_minute": round(len(cuts) / minutes, 1)}
    fast = _busiest_windows(cuts, CUT_WINDOW, 2.5, minimum=4)
    if fast:
        scene["fast_cutting"] = [{"start": w["start"], "end": w["end"], "cuts": w["count"]} for w in fast]
    out["scene_changes"] = scene

    spoken = sum(s["end"] - s["start"] for s in segments) or 1.0
    speech = {"avg_words_per_second": round(sum((s.get("wps") or 0) * (s["end"] - s["start"]) for s in segments) / spoken, 1)}
    fast_lines = sorted(
        ((i, s) for i, s in enumerate(segments) if (s.get("wps") or 0) >= 4.5 and s["end"] - s["start"] >= 1.5),
        key=lambda x: -x[1]["wps"])[:5]
    if fast_lines:
        speech["fast_segments"] = [f"seg_{i}" for i, _ in fast_lines]
    out["speech"] = speech

    pauses = sorted(((i, s) for i, s in enumerate(segments) if (s.get("pause_before") or 0) >= 1.5),
                    key=lambda x: -x[1]["pause_before"])[:3]
    if pauses:
        out["longest_silences"] = [{"before": f"seg_{i}", "at": round(s["start"], 1), "seconds": s["pause_before"]}
                                   for i, s in pauses]

    labels = [s.get("speaker") for s in segments]
    if any(l not in (None, "unknown") for l in labels):
        change_times = [segments[i]["start"] for i in range(1, len(segments))
                        if labels[i] != labels[i - 1] and labels[i] not in (None, "unknown")
                        and labels[i - 1] not in (None, "unknown")]
        exchange = {"speaker_changes": len(change_times)}
        rapid = _busiest_windows(change_times, EXCHANGE_WINDOW, 5.0, minimum=4)
        if rapid:
            exchange["rapid_exchanges"] = [{"start": w["start"], "end": w["end"], "changes": w["count"]} for w in rapid]
        out["conversation"] = exchange

    def expressive(key):
        found = sorted(
            ((i, (s.get(key) or {})) for i, s in enumerate(segments)),
            key=lambda x: -float(x[1].get("intensity", 0.0)))
        return [{"segment": f"seg_{i}", "emotion": f["emotion"], "intensity": f["intensity"]}
                for i, f in found if f.get("emotion") not in (None, "neutral") and f.get("intensity", 0) >= 0.5][:5]

    faces, listeners = expressive("face"), expressive("listener")
    if faces:
        out["strongest_expressions"] = faces
    if listeners:
        out["listener_reactions"] = listeners

    if heatmap:
        top = sorted(heatmap, key=lambda b: -b["value"])[:3]
        out["most_replayed"] = [{"start": round(b["start_time"], 1), "end": round(b["end_time"], 1),
                                 "score": round(b["value"], 2)} for b in sorted(top, key=lambda b: b["start_time"])]
    return out
