import ipaddress
import socket
from urllib.parse import urlparse


def is_public_url(url: str) -> bool:
    """True only for http(s) URLs whose host resolves to public internet addresses."""
    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, ValueError):
        return False

    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:          # private, loopback, link-local, reserved
            return False
    return bool(infos)