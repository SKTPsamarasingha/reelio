"""Face-following crop: where the crop window should sit over time for a vertical clip."""
import logging

from app.pipeline.media import open_video, read_frame_at

log = logging.getLogger(__name__)

SAMPLE_FPS = 2.0
MAX_KEYS = 90            # cap on keyframes, so the ffmpeg expression stays short
MIN_FACE_AREA = 0.004    # wide shots have small faces
CUT_JUMP = 0.18          # a bigger jump (fraction of frame width) is treated as a camera cut
SMOOTH_WINDOW = 5


def _moving_average(vals, window):
    half = window // 2
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - half), min(len(vals), i + half + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def _pick_center(faces, crop_frac):
    """faces: [(cx, width, area)] as fractions of the frame. Returns the x to centre on."""
    if not faces:
        return None
    left = min(cx - w / 2 for cx, w, _ in faces)
    right = max(cx + w / 2 for cx, w, _ in faces)
    if right - left <= crop_frac * 0.85:           # everyone fits in the crop: keep them all
        return (left + right) / 2
    return max(faces, key=lambda f: f[2])[0]        # otherwise follow the biggest face


class FaceTrack:
    def __init__(self, samples, src_w, src_h):
        self.samples = samples          # [(t, [(cx, width, area), ...]), ...], t relative to the clip start
        self.src_w, self.src_h = src_w, src_h

    def keyframes(self, crop_frac):
        """[(t, travel 0-1)] for a crop whose width is crop_frac of the frame, or None."""
        if crop_frac >= 0.999 or not self.samples:
            return None
        centers = [_pick_center(f, crop_frac) for _, f in self.samples]
        if all(c is None for c in centers):
            return None

        last = next(c for c in centers if c is not None)
        filled = []
        for c in centers:
            last = c if c is not None else last
            filled.append(last)

        runs = [[0]]                                    # split at camera cuts
        for i in range(1, len(filled)):
            if abs(filled[i] - filled[i - 1]) > CUT_JUMP:
                runs.append([i])
            else:
                runs[-1].append(i)

        keys = []
        for run in runs:
            vals = _moving_average([filled[i] for i in run], SMOOTH_WINDOW)
            first_t = self.samples[run[0]][0]
            if keys:                                    # hold the old position until just before the cut
                hold_t = first_t - 0.04
                if hold_t > keys[-1][0]:
                    keys.append((hold_t, keys[-1][1]))
            for idx, v in zip(run, vals):
                frac = min(1.0, max(0.0, (v - crop_frac / 2) / (1 - crop_frac)))
                keys.append((self.samples[idx][0], frac))
        return keys


def track_faces(video_path, start, end):
    """Sample faces across [start, end]. Returns a FaceTrack, or None if anything goes wrong."""
    import cv2

    from app.pipeline.faces import detect_faces

    try:
        with open_video(video_path) as cap:
            src_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            src_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            duration = max(end - start, 0.1)
            step = max(1.0 / SAMPLE_FPS, duration / MAX_KEYS)

            samples, t = [], 0.0
            while t <= duration:
                frame = read_frame_at(cap, start + t)
                faces = []
                if frame is not None:
                    faces = [(f.cx, f.width, f.area) for f in detect_faces(frame, MIN_FACE_AREA)]
                samples.append((round(t, 3), faces))
                t += step
        return FaceTrack(samples, src_w, src_h)
    except Exception:
        log.exception("face tracking failed; falling back to a centred crop")
        return None
