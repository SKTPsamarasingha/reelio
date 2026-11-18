"""Download a video from a link (local testing only) and reuse an earlier copy when we have one."""
import json
import os

import yt_dlp

from app.config import get_settings
from app.core import cache
from app.pipeline.media import ffmpeg_exe
from app.pipeline.progress import ytdlp_hook
from pathlib import Path

# Resolves to D:\Projects\reelio\backend\cookies.txt
COOKIES_FILE = Path(__file__).resolve().parents[2] / "cookies.txt"


def _video_key(video_id: str) -> str:
    return cache.key("video", video_id)


def _remember(video_id: str, path: str) -> None:
    cache.cache_set(
        _video_key(video_id),
        json.dumps({"file": os.path.basename(path), "size": os.path.getsize(path)}),
        get_settings().video_cache_days * 86400,
    )


def _usable(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def find_cached_video(video_id: str, output_folder: str = "video") -> str | None:
    """Path of an already-downloaded video, or None. Redis is a hint; the disk is the final authority."""
    raw = cache.cache_get(_video_key(video_id))
    if raw:
        try:
            candidate = os.path.join(output_folder, json.loads(raw)["file"])
        except (ValueError, KeyError, TypeError):
            candidate = None
        if candidate and _usable(candidate):
            _remember(video_id, candidate)  # refresh the expiry
            return candidate

    path = os.path.join(output_folder, f"{video_id}.mp4")
    if _usable(path):
        _remember(video_id, path)
        return path
    return None


def _download(url: str, output_folder: str) -> str:
    height = get_settings().video_max_height
    opts = {
        "format": f"bv*[height<={height}]+ba/b[height<={height}]/b",
        "outtmpl": os.path.join(output_folder, "%(id)s.%(ext)s"),
        "overwrites": True,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [ytdlp_hook()],
        "ffmpeg_location": ffmpeg_exe(),
        # "cookiesfrombrowser": ("chrome",),
        # "cookiefile": "cookies.txt",
    }
    if COOKIES_FILE.is_file():
        opts["cookiefile"] = str(COOKIES_FILE)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return os.path.splitext(ydl.prepare_filename(info))[0] + ".mp4"


def download_video(url: str, output_folder: str = "video", video_id: str | None = None) -> str:
    """Download, or return the saved copy. Pass video_id to enable reuse and the cross-worker lock."""
    os.makedirs(output_folder, exist_ok=True)
    if not video_id:
        return _download(url, output_folder)

    hit = find_cached_video(video_id, output_folder)
    if hit:
        return hit

    with cache.lock(f"download:{video_id}"):
        hit = find_cached_video(video_id, output_folder)  # another worker may have just finished
        if hit:
            return hit
        path = _download(url, output_folder)
        _remember(video_id, path)
        return path
