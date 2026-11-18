"""YouTube video id from a link, without any network call."""
import re
from urllib.parse import parse_qs, urlparse

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}


def extract_video_id(url: str) -> str | None:
    """The 11-character id, or None for other sites and malformed links."""
    try:
        p = urlparse((url or "").strip())
    except ValueError:
        return None
    host = (p.hostname or "").lower()
    if host not in _HOSTS:
        return None

    if host.endswith("youtu.be"):
        candidate = p.path.lstrip("/").split("/")[0]
    elif p.path == "/watch":
        candidate = parse_qs(p.query).get("v", [""])[0]
    else:
        parts = [x for x in p.path.split("/") if x]
        candidate = parts[1] if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v") else ""

    return candidate if _ID.match(candidate) else None
