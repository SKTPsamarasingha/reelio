"""Title, description and tags used as hints for transcription and analysis.

Everything here is optional: safe_video_context() never raises, and a video's metadata is saved next to
the video so it only has to be fetched successfully once.
"""
import json
import logging
import re
import time
import urllib.parse
import urllib.request

import yt_dlp

from app.config import get_settings

log = logging.getLogger(__name__)

META_FRESH_SECONDS = 7 * 86400      # re-fetch after a week; the replay heatmap changes slowly

# Capitalised words that are usually not names
COMMON = {
    "The", "This", "That", "These", "Those", "And", "But", "For", "With", "From",
    "Subscribe", "Follow", "Like", "Watch", "Video", "Official", "Channel",
    "Instagram", "Twitter", "Facebook", "Youtube", "Click", "Link", "Music",
    "Credit", "Credits", "Copyright", "All", "Rights", "Reserved", "Part",
}


def _empty(video_id=None) -> dict:
    return {"id": video_id, "duration": None, "title": "", "context": "", "vocabulary": [],
            "categories": [], "tags": [], "heatmap": None}


def _names(text: str) -> list[str]:
    """Words capitalised in the middle of a sentence are likely names or terms."""
    found = []
    for sentence in re.split(r"[.!?\n]+", text):
        words = re.findall(r"[A-Za-z][A-Za-z'\-]+", sentence)
        found += [w for w in words[1:] if w[0].isupper() and w not in COMMON and len(w) > 2]
    return found


def _dedupe(terms, limit) -> list[str]:
    seen, out = set(), []
    for term in terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return out[:limit]


def fetch_video_context(url: str, max_terms: int = 30) -> dict:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "socket_timeout": 15}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title") or ""
    description = re.sub(r"https?://\S+", "", (info.get("description") or "")[:1500]).strip()
    tags = info.get("tags") or []
    first_sentence = re.split(r"(?<=[.!?])\s", description)[0] if description else ""

    return {
        "id": info.get("id"),
        "duration": info.get("duration"),
        "title": title,
        "context": f"{title}. {first_sentence}".strip()[:300],
        "vocabulary": _dedupe([*_names(f"{title}. {description}"), *tags], max_terms),
        "categories": info.get("categories") or [],
        "tags": tags,
        "heatmap": info.get("heatmap"),
    }


def _oembed(url: str) -> dict:
    """Just the title and channel name. Lightweight, and usually not bot-checked."""
    query = urllib.parse.urlencode({"url": url, "format": "json"})
    with urllib.request.urlopen(f"https://www.youtube.com/oembed?{query}", timeout=6) as response:
        return json.load(response)


def _sidecar(video_id: str):
    return get_settings().data_dir / "videos" / f"{video_id}.meta.json"


def _read_sidecar(video_id: str):
    try:
        path = _sidecar(video_id)
        return json.loads(path.read_text(encoding="utf-8")), time.time() - path.stat().st_mtime
    except (OSError, ValueError):
        return None, None


def _write_sidecar(video_id: str, meta: dict) -> None:
    try:
        path = _sidecar(video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta), encoding="utf-8")
    except OSError:
        pass


def safe_video_context(url: str, video_id: str | None = None) -> dict:
    """Never raises. Order: fresh saved copy, live yt-dlp, stale saved copy, oEmbed, empty."""
    saved, age = _read_sidecar(video_id) if video_id else (None, None)
    if saved and age < META_FRESH_SECONDS:
        return saved

    try:
        meta = fetch_video_context(url)
        if video_id or meta.get("id"):
            _write_sidecar(video_id or meta["id"], meta)
        return meta
    except Exception as exc:
        log.warning("yt-dlp metadata unavailable (%s)", exc.__class__.__name__)

    if saved:                       # old but real, and it still has the heatmap
        return saved

    meta = _empty(video_id)
    try:
        data = _oembed(url)
        title, author = data.get("title", ""), data.get("author_name", "")
        meta["title"] = title
        meta["context"] = f"{title}. {author}".strip(". ")[:300]
        meta["vocabulary"] = _dedupe(
            [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", f"{title} {author}")
             if w[0].isupper() and w not in COMMON and len(w) > 2], 20)
    except Exception:
        log.warning("oEmbed fallback unavailable too; continuing without video metadata")
    return meta
