import hashlib
import logging

from app.config import get_settings
from app.core import storage
from app.core.errors import UserFacingError
from app.core.urlsafe import is_public_url
from app.worker.queue import JobCanceled

log = logging.getLogger(__name__)


def _meta(duration=None):
    return {"id": None, "duration": duration, "title": "", "context": "", "vocabulary": [],
            "categories": [], "tags": [], "heatmap": None}


def video_seconds(path) -> float | None:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps, frames = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return frames / fps if fps and frames else None
    finally:
        cap.release()


def prepare_source(source_type, source_key, url, sp, job_id):
    """Returns (video_path, meta). Uploads come from our own Cloudinary account."""
    s = get_settings()
    cache_dir = s.data_dir / "videos"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if source_type == "UPLOAD":
        try:
            info = storage.inspect_video(source_key)
        except Exception as exc:
            raise UserFacingError("Couldn't reach storage. Please try again.") from exc
        if info is None:
            raise UserFacingError("We couldn't find the uploaded video. Please upload it again.")

        key = hashlib.sha1(f"{source_key}:{info['version']}".encode()).hexdigest()[:16]
        path = cache_dir / f"up_{key}.{info['format']}"
        if path.exists() and path.stat().st_size > 0:
            sp.at(1.0, "Using the saved copy of this video")
        else:
            sp.at(0.05, "Downloading your video")
            try:
                storage.download_source(source_key, info["version"], info["format"], path,
                                        s.max_upload_mb * 1024 * 1024,
                                        on_progress=lambda f: sp.at(0.05 + 0.9 * f, "Downloading your video"))
            except JobCanceled:
                raise
            except Exception as exc:
                raise UserFacingError("Couldn't download the uploaded video. Please try again.") from exc
        return str(path), _meta(info["duration"] or None)

    # ---- link sources (local testing only) ----
    from app.pipeline.downloader import download_video, find_cached_video
    from app.pipeline.metadata import safe_video_context
    from app.pipeline.urls import extract_video_id

    if not s.allow_url_sources or not is_public_url(url):
        raise UserFacingError("That link can't be used. Please upload your video file.")

    video_id = extract_video_id(url)
    cached = find_cached_video(video_id, str(cache_dir)) if video_id else None
    meta = safe_video_context(url, video_id)
    video_id = video_id or meta.get("id") or job_id
    if cached:
        sp.at(1.0, "Using the saved copy of this video")
        return cached, meta
    sp.at(0.1, "Downloading video")
    try:
        return download_video(url, output_folder=str(cache_dir), video_id=video_id), meta
    except JobCanceled:
        raise
    except Exception as exc:
        if "not a bot" in str(exc) or "Sign in to confirm" in str(exc):
            raise UserFacingError("YouTube is blocking automated downloads right now. Please upload the file instead.") from exc
        raise UserFacingError("Couldn't download that video. Please upload the file instead.") from exc