"""Camera cuts, found by comparing colour histograms of sampled frames."""
from app.pipeline.media import open_video
from app.pipeline.progress import Bar, ok

CUT_THRESHOLD = 0.45   # Bhattacharyya distance: 0 = identical, 1 = completely different
SAMPLE_FPS = 10        # frames analysed per second of video
MIN_GAP = 0.3          # ignore cuts closer together than this (flashes, explosions)
LEAD = 0.3             # a cut this close BEFORE a segment still counts for it


def _hist(frame):
    import cv2
    small = cv2.resize(frame, (160, max(int(160 * frame.shape[0] / frame.shape[1]), 1)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def detect_cuts(video_path) -> list[float]:
    """Times (seconds) of hard camera cuts."""
    import cv2
    cuts, prev, idx = [], None, 0
    with open_video(video_path) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(int(round(fps / SAMPLE_FPS)), 1)
        bar = Bar("scene cuts", total)

        while True:
            if idx % step == 0:
                got, frame = cap.read()
            else:
                got, frame = cap.grab(), None      # skip the colour conversion for frames we don't analyse
            if not got:
                break

            if frame is not None:
                hist = _hist(frame)
                if prev is not None:
                    t = idx / fps
                    if cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA) > CUT_THRESHOLD \
                            and (not cuts or t - cuts[-1] > MIN_GAP):
                        cuts.append(round(t, 2))
                prev = hist

            idx += 1
            if idx % 300 == 0:
                bar.update(min(idx, total))
        bar.finish()

    ok(f"Scene cuts found: {len(cuts)}")
    return cuts


def add_scene_signals(segments, cuts: list[float]):
    """Adds scene_cut (bool) and cut_count (int) to each segment."""
    for seg in segments:
        inside = [c for c in cuts if seg["start"] - LEAD <= c <= seg["end"]]
        seg["scene_cut"] = bool(inside)
        seg["cut_count"] = len(inside)
    return segments
