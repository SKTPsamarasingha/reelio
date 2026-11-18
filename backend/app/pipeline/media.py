"""Shared helpers for video and audio: one ffmpeg, one audio decode, one way to open a video."""
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16_000   # Hz, what Whisper and pyannote expect


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """The single ffmpeg binary used everywhere (decoding, cutting, downloads)."""
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio(video_path) -> np.ndarray:
    """Decode the audio track once: 16 kHz mono float32. Streams through a pipe, no temp files."""
    cmd = [
        ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace")[-300:]
        raise RuntimeError(f"Could not read the audio track. {detail}".strip())
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps > 0 and self.frames > 0 else 0.0


@contextmanager
def open_video(path):
    """Yield an OpenCV capture and always release it."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        yield cap
    finally:
        cap.release()


def probe_video(path) -> VideoInfo:
    import cv2
    with open_video(path) as cap:
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0),
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )


def read_frame_at(cap, seconds: float):
    """The frame at `seconds`, or None if it can't be read."""
    import cv2
    cap.set(cv2.CAP_PROP_POS_MSEC, max(seconds, 0.0) * 1000.0)
    got, frame = cap.read()
    return frame if got else None


def shrink(frame, max_side: int):
    """Scale down so the longest side is at most max_side. Never upscales."""
    import cv2
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)
