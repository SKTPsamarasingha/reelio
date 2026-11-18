"""Per-segment audio features: loudness, pace, pauses and crowd reactions."""
from dataclasses import dataclass

import numpy as np

from app.pipeline.media import SAMPLE_RATE
from app.pipeline.progress import Bar, ok

WINDOW, HOP = 1.0, 0.5          # seconds
BASELINE_SECONDS = 30.0         # "normal" loudness is the median of this much surrounding audio
REACTION_MAX = 4.0              # look this far after a line for a crowd reaction


@dataclass(frozen=True)
class Loudness:
    times: np.ndarray     # window centres, seconds
    db: np.ndarray        # loudness per window
    level: np.ndarray     # 0-1, relative to this video
    spike: np.ndarray     # 0-1, how far above the surrounding baseline (steady music cancels out)

    @property
    def empty(self) -> bool:
        return self.times.size == 0


def scale01(values, lo_pct=5, hi_pct=95, flat=0.0) -> np.ndarray:
    """Scale to 0-1 using percentiles, so one huge spike doesn't squash everything else."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    if hi - lo < 1e-9:
        return np.full(values.shape, flat)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def loudness_curves(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Loudness:
    win, step = int(sr * WINDOW), int(sr * HOP)
    if len(audio) < win:
        empty = np.array([])
        return Loudness(empty, empty, empty, empty)

    n = 1 + (len(audio) - win) // step
    cs = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
    starts = np.arange(n) * step
    db = 10 * np.log10((cs[starts + win] - cs[starts]) / win + 1e-12)

    half = max(int(BASELINE_SECONDS / HOP / 2), 1)
    baseline = np.array([np.median(db[max(0, i - half): i + half + 1]) for i in range(len(db))])
    return Loudness((starts + win / 2) / sr, db, scale01(db), scale01(np.maximum(db - baseline, 0.0)))


def _peak(curves: Loudness, a: float, b: float) -> float:
    if curves.empty:
        return 0.0
    part = curves.spike[np.searchsorted(curves.times, a): np.searchsorted(curves.times, b)]
    return float(part.max()) if part.size else 0.0


def add_audio_signals(segments, audio: np.ndarray, curves: Loudness):
    """Adds energy (0-1), wps, pause_before (s), peak (0-1, loudest spike inside the line)
    and reaction_after (0-1, loud burst in the gap after the line, the laughter/applause shape)."""
    sr = SAMPLE_RATE
    bar = Bar("audio signals", len(segments) or 1)

    dbs = []
    for seg in segments:
        chunk = audio[int(seg["start"] * sr): int(seg["end"] * sr)]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        dbs.append(20 * np.log10(rms + 1e-9))
    energy = scale01(dbs, flat=0.5)

    for i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        seg["energy"] = round(float(energy[i]), 2)

        words = seg.get("words") or []
        if words:
            span, count = words[-1]["e"] - words[0]["s"], len(words)
        else:
            span, count = end - start, len(seg["text"].split())
        seg["wps"] = round(count / span, 1) if span > 0.05 else 0.0

        prev_end = segments[i - 1]["end"] if i else 0.0
        seg["pause_before"] = round(max(start - prev_end, 0.0), 2)

        seg["peak"] = round(_peak(curves, start, end), 2)

        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else end + REACTION_MAX
        gap_a, gap_b = end + 0.1, min(next_start, end + REACTION_MAX)
        seg["reaction_after"] = round(_peak(curves, gap_a, gap_b), 2) if gap_b - gap_a >= 0.4 else 0.0
        bar.update(i + 1)

    bar.finish()
    ok(f"Audio signals added to {len(segments)} segments")
    return segments
