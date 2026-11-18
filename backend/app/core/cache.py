import logging
import time
import uuid
from contextlib import contextmanager

import redis

from app.config import get_settings

log = logging.getLogger(__name__)
_client = None
_failed_at = 0.0


def get_redis():
    """Return a Redis client, or None if Redis isn't configured or is unreachable.
    Callers must treat None as 'no cache' and carry on."""
    global _client, _failed_at
    url = get_settings().redis_url
    if not url:
        return None
    if _client is not None:
        return _client
    if time.monotonic() - _failed_at < 30:       # don't retry on every call while it's down
        return None
    try:
        client = redis.Redis.from_url(
            url, decode_responses=True, socket_connect_timeout=3, socket_timeout=3
        )
        client.ping()
        _client = client
        return _client
    except Exception as exc:
        _failed_at = time.monotonic()
        log.warning("Redis unavailable, continuing without cache: %s", exc.__class__.__name__)
        return None


def key(*parts: str) -> str:
    return ":".join(["reelio", f"v{get_settings().pipeline_version}", *parts])


def cache_get(k: str):
    r = get_redis()
    if r is None:
        return None
    try:
        return r.get(k)
    except Exception:
        return None


def cache_set(k: str, value: str, ttl_seconds: int) -> None:
    r = get_redis()
    if r is None:
        return
    try:
        r.set(k, value, ex=ttl_seconds)
    except Exception:
        pass


@contextmanager
def lock(name: str, ttl: int = 900, wait: int = 600):
    """Cross-worker lock. Without Redis it does nothing (single worker is fine).
    If another worker holds it, waits up to `wait` seconds, then proceeds anyway."""
    r = get_redis()
    if r is None:
        yield
        return

    k, token = key("lock", name), uuid.uuid4().hex
    deadline = time.monotonic() + wait
    held = False
    try:
        while True:
            try:
                held = bool(r.set(k, token, nx=True, ex=ttl))
            except Exception:
                held = False
                break
            if held or time.monotonic() >= deadline:
                break
            time.sleep(3)
        yield
    finally:
        if held:
            try:
                if r.get(k) == token:        # only release our own lock
                    r.delete(k)
            except Exception:
                pass