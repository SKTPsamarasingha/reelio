"""Snap model-chosen clips to sentence boundaries and enforce the requested length range."""
PAD_BEFORE, PAD_AFTER = 0.15, 0.25
OVERLAP_LIMIT = 0.5


def _nearest(values, target):
    return min(range(len(values)), key=lambda k: abs(values[k] - target))


def fit_clips(clips, segments, min_len, max_len, duration):
    """Snap to transcript segment boundaries, enforce min/max length, drop overlaps."""
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]
    kept = []

    for clip in clips:
        s = min(max(float(clip["start_timestamp"]), 0.0), duration)
        e = min(max(float(clip["end_timestamp"]), 0.0), duration)
        if e <= s:
            continue

        if segments:
            i = _nearest(starts, s)
            j = max(_nearest(ends, e), i)
            while j > i and ends[j] - starts[i] > max_len:          # too long: drop trailing sentences
                j -= 1
            while ends[j] - starts[i] < min_len:                    # too short: add neighbours
                if j + 1 < len(segments) and ends[j + 1] - starts[i] <= max_len:
                    j += 1
                elif i > 0 and ends[j] - starts[i - 1] <= max_len:
                    i -= 1
                else:
                    break
            s, e = starts[i], ends[j]

        s, e = max(0.0, s - PAD_BEFORE), min(duration, e + PAD_AFTER)
        if e - s > max_len:
            e = s + max_len
        if e - s < min_len:
            e = min(duration, s + min_len)
            if e - s < min_len:
                s = max(0.0, e - min_len)

        if any(min(e, k["end_timestamp"]) - max(s, k["start_timestamp"]) > OVERLAP_LIMIT for k in kept):
            continue
        kept.append({**clip, "start_timestamp": round(s, 2), "end_timestamp": round(e, 2)})
    return kept
