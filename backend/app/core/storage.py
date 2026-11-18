import logging
from urllib.parse import urlparse

import cloudinary
import cloudinary.api
import cloudinary.uploader
from cloudinary.utils import cloudinary_url as build_url

from app.config import get_settings
from pathlib import Path
import re
import time, uuid
# ... your other imports ...
from cloudinary.utils import api_sign_request

log = logging.getLogger(__name__)
_configured = False


def is_configured() -> bool:
    return bool(get_settings().cloudinary_url)


def _configure() -> None:
    global _configured
    if _configured:
        return
    url = get_settings().cloudinary_url
    if not url:
        raise RuntimeError("CLOUDINARY_URL is not set in backend/.env")
    p = urlparse(url)
    cloudinary.config(
        cloud_name=p.netloc.rsplit("@", 1)[-1],
        api_key=p.username,
        api_secret=p.password,
        secure=True,
    )
    _configured = True


def _folder() -> str:
    return get_settings().cloudinary_folder.strip("/")


def _public_id(key: str) -> str:
    last = key.rsplit("/", 1)[-1]
    if "." in last:
        key = key.rsplit(".", 1)[0]
    folder = _folder()
    return f"{folder}/{key}" if folder else key


class ObjectStorage:
    def __init__(self):
        _configure()

    def upload_jpeg(self, data: bytes, key: str) -> str:
        """Used by frames.py for contact sheets. Returns the URL."""
        res = cloudinary.uploader.upload(
            data,
            public_id=_public_id(f"sheets/{key}"),
            resource_type="image",
            overwrite=True,
            invalidate=True,
        )
        return res["secure_url"]

    def upload_clip(self, path: str, key: str) -> str:
        """Upload a rendered clip. Returns the public_id, which we store in the database."""
        res = cloudinary.uploader.upload(
            path,
            public_id=_public_id(key),
            resource_type="video",
            overwrite=True,
            invalidate=True,
        )
        return res["public_id"]

    def delete_job_assets(self, job_id: str) -> None:
        """Best effort: remove every clip and sheet that belongs to a job."""
        folder = _folder()
        prefixes = [
            ("video", f"{folder}/jobs/{job_id}/"),
            ("image", f"{folder}/sheets/{job_id}/"),
        ]
        for resource_type, prefix in prefixes:
            try:
                cloudinary.api.delete_resources_by_prefix(
                    prefix, resource_type=resource_type, invalidate=True
                )
            except Exception:
                log.warning("could not delete %s assets with prefix %s", resource_type, prefix)


def clip_urls(public_id: str) -> tuple[str | None, str | None]:
    """(play_url, download_url) for a stored clip, or (None, None) if storage isn't configured."""
    if not is_configured():
        return None, None
    _configure()
    play, _ = build_url(public_id, resource_type="video", format="mp4", secure=True)
    download, _ = build_url(
        public_id, resource_type="video", format="mp4", secure=True, flags="attachment"
    )
    return play, download


def delete_job_assets(job_id: str) -> None:
    if is_configured():
        ObjectStorage().delete_job_assets(job_id)


def delete_clip_asset(public_id: str) -> None:
    """Best effort: remove a single clip file from Cloudinary."""
    if not public_id or not is_configured():
        return
    try:
        _configure()
        cloudinary.uploader.destroy(public_id, resource_type="video", invalidate=True)
    except Exception:
        log.warning("could not delete clip asset %s", public_id)


_ID_CHARS = re.compile(r"^[A-Za-z0-9_\-./]+$")


def is_video_public_id(public_id: str) -> bool:
    """Only accept IDs inside our own videos folder."""
    folder = _folder()
    prefix = f"{folder}/videos/" if folder else "videos/"
    return (
        public_id.startswith(prefix) and len(public_id) > len(prefix) and len(public_id) <= 200
        and ".." not in public_id and bool(_ID_CHARS.match(public_id))
    )


def inspect_video(public_id: str) -> dict | None:
    """Real facts about an uploaded video from Cloudinary, or None if it doesn't exist.
    Raises on network/auth problems, so callers can tell 'missing' from 'couldn't check'."""
    _configure()
    try:
        r = cloudinary.api.resource(public_id, resource_type="video")
    except cloudinary.exceptions.NotFound:
        return None
    return {
        "duration": float(r.get("duration") or 0),     # 0 = unknown
        "bytes": int(r.get("bytes") or 0),
        "format": r.get("format") or "mp4",
        "version": r.get("version"),
    }


def download_source(public_id, version, fmt, dest: Path, max_bytes: int, on_progress=None) -> None:
    """Download the original upload. The URL is built here from our own account, never taken from the client."""
    import requests

    _configure()
    url, _ = build_url(public_id, resource_type="video", format=fmt, version=version, secure=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length") or 0)
            if total and total > max_bytes:
                raise ValueError("file too large")
            done = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(256 * 1024):
                    done += len(chunk)
                    if done > max_bytes:
                        raise ValueError("file too large")
                    f.write(chunk)
                    if on_progress and total:
                        on_progress(done / total)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


ALLOWED_FORMATS = "mp4,mov,webm,mkv,m4v"


def sign_upload(user_id: str) -> dict:
    """Signed parameters that let the browser upload one video, only inside this user's folder."""
    _configure()
    cfg = cloudinary.config()
    folder = _folder()
    public_id = f"{folder + '/' if folder else ''}videos/{user_id}/{uuid.uuid4().hex}"
    timestamp = int(time.time())
    params = {"public_id": public_id, "timestamp": timestamp, "allowed_formats": ALLOWED_FORMATS}
    return {
        "uploadUrl": f"https://api.cloudinary.com/v1_1/{cfg.cloud_name}/video/upload",
        "apiKey": cfg.api_key,
        "timestamp": timestamp,
        "publicId": public_id,
        "allowedFormats": ALLOWED_FORMATS,
        "signature": api_sign_request(params, cfg.api_secret),
    }