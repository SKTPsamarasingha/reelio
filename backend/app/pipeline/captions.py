"""Burned-in captions and the hook banner, written as an ASS subtitle file per clip."""
import os
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import BACKEND_DIR, get_settings
from app.options import LAYOUTS

HOOK_SECONDS = 3.5
HOLD_AFTER_WORD = 0.25
MIN_EVENT = 0.08
MAX_GAP = 0.9


@dataclass(frozen=True)
class CaptionStyle:
    size: float                 # font size as a fraction of the frame height
    primary: str                # BBGGRR
    highlight: str | None       # BBGGRR of the word being spoken, or None
    outline: float              # outline (or box padding) as a fraction of the height
    uppercase: bool
    strip_punct: bool
    words_per_chunk: int
    box: str | None             # 8-digit ASS colour of a background box, or None


STYLES = {
    "hormozi": CaptionStyle(0.052, "FFFFFF", "00FFFF", 0.0042, True, True, 3, None),
    "clean":   CaptionStyle(0.038, "FFFFFF", None, 0.012, False, False, 7, "&H99000000"),
}

_ALWAYS = re.compile(r"[{}\\]")
_PUNCT = re.compile(r"[.,!?;:\"“”«»()\[\]]")


def fonts_dir() -> str | None:
    """Folder libass should look in. Our own fonts first, then the system's."""
    ours = BACKEND_DIR / "assets" / "fonts"
    if ours.is_dir() and (any(ours.glob("*.ttf")) or any(ours.glob("*.otf"))):
        return str(ours)
    for d in (r"C:\Windows\Fonts", "/usr/share/fonts", "/Library/Fonts"):
        if os.path.isdir(d):
            return d
    return None


def _t(sec: float) -> str:
    cs = int(round(max(sec, 0.0) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _clean(word: str, style: CaptionStyle) -> str:
    w = _ALWAYS.sub("", word)
    if style.strip_punct:
        w = _PUNCT.sub("", w)
    w = w.strip()
    return w.upper() if style.uppercase else w


def _clip_words(segments, start, end):
    out = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        for w in seg.get("words") or []:
            if w.get("w") and start - 0.05 <= w["s"] < end:
                out.append({"w": w["w"], "s": max(w["s"], start) - start,
                            "e": min(max(w["e"], w["s"]), end) - start})
    return out


def _caption_events(words, style):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= style.words_per_chunk or w["s"] - cur[-1]["e"] > MAX_GAP):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)

    events = []
    for ci, chunk in enumerate(chunks):
        texts = [_clean(w["w"], style) for w in chunk]
        chunk_end = chunk[-1]["e"] + HOLD_AFTER_WORD
        if ci + 1 < len(chunks):
            chunk_end = min(chunk_end, chunks[ci + 1][0]["s"])

        if style.highlight is None:
            line = " ".join(t for t in texts if t)
            if line:
                events.append((chunk[0]["s"], max(chunk_end, chunk[0]["s"] + MIN_EVENT), line))
            continue

        for k, w in enumerate(chunk):
            if not texts[k]:
                continue
            t0 = w["s"]
            t1 = max(chunk[k + 1]["s"] if k + 1 < len(chunk) else chunk_end, t0 + MIN_EVENT)
            line = " ".join(
                f"{{\\1c&H{style.highlight}&}}{tx}{{\\1c&H{style.primary}&}}" if i == k else tx
                for i, tx in enumerate(texts) if tx
            )
            events.append((t0, t1, line))
    return events


def build_ass(path, segments, clip_start, clip_end, layout, caption_style, hook_text="") -> bool:
    """Write an .ass file for one clip. Returns False when there is nothing to show."""
    lay = LAYOUTS[layout]
    w, h = lay.width, lay.height
    font = get_settings().caption_font
    style = STYLES.get(caption_style)
    margin_lr = int(w * 0.06)

    events = []
    if style:
        events += [("Caption", 0, a, b, text)
                   for a, b, text in _caption_events(_clip_words(segments, clip_start, clip_end), style)]
    hook_text = re.sub(r"[{}\\]", "", (hook_text or "")).strip()[:90]
    if hook_text:
        events.append(("Hook", 1, 0.0, min(HOOK_SECONDS, clip_end - clip_start), "{\\fad(200,200)}" + hook_text))
    if not events:
        return False

    styles = []
    if style:
        box = style.box
        styles.append(
            f"Style: Caption,{font},{style.size * h:.0f},&H00{style.primary},&H000000FF,"
            f"{box or '&H00000000'},{box or '&H00000000'},-1,0,0,0,100,100,0,0,{3 if box else 1},"
            f"{style.outline * h:.1f},0,2,{margin_lr},{margin_lr},{int(h * lay.caption_margin)},1"
        )
    styles.append(
        f"Style: Hook,{font},{0.04 * h:.0f},&H00000000,&H000000FF,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,"
        f"{0.012 * h:.1f},0,8,{margin_lr},{margin_lr},{int(h * lay.hook_margin)},1"
    )

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        *styles, "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for name, layer, a, b, text in events:
        lines.append(f"Dialogue: {layer},{_t(a)},{_t(b)},{name},,0,0,0,,{text}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True
