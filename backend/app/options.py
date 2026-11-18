import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    caption_margin: float   # caption distance from the bottom edge, as a fraction of the height
    hook_margin: float      # hook banner distance from the top edge, as a fraction of the height


LAYOUTS = {
    "9:16": Layout(1080, 1920, 0.24, 0.11),   # keeps captions clear of Reels/Shorts/TikTok buttons
    "4:5":  Layout(1080, 1350, 0.14, 0.08),
    "1:1":  Layout(1080, 1080, 0.10, 0.06),
    "16:9": Layout(1920, 1080, 0.09, 0.06),
}
FRAMINGS = ("fill", "fit")
CAPTION_STYLES = ("none", "hormozi", "clean")
HOOK_MODES = ("none", "text_banner")

AI_PICKS = {
    "best_overall": "Pick the strongest, most self-contained moments overall.",
    "viral_hooks": (
        "Pick moments that open with a strong hook (a bold claim, a surprising fact, a question or a "
        "cliffhanger in the first 3 seconds) and pay off quickly. Avoid slow openings."
    ),
    "funny_moments": "Pick the funniest moments. Include the full setup, the punchline and any laughter or reaction.",
    "key_insights": "Pick the most useful self-contained insights, tips or explanations a viewer could remember or act on.",
    "emotional_moments": "Pick the most emotional moments: strong feelings, vulnerable stories, big reactions.",
    "best_quotes": "Pick the most quotable lines and short statements that stand on their own.",
}
AI_PICK_ALIASES = {
    "auto": "best_overall", "best": "best_overall", "hooks": "viral_hooks", "funny": "funny_moments",
    "insights": "key_insights", "emotional": "emotional_moments", "quotes": "best_quotes",
}

# Used only when an older job is missing a field.
DEFAULTS = {
    "layout": "9:16", "framing": "fill", "ai_picks": "best_overall",
    "caption_style": "none", "hook_mode": "none", "clip_min": 15, "clip_max": 60,
}


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def resolve_ai_pick(value: str | None) -> str | None:
    key = normalize_key(value or "best_overall")
    key = AI_PICK_ALIASES.get(key, key)
    return key if key in AI_PICKS else None


def target_clip_count(duration: float, min_sec: float, max_sec: float, cap: int, requested: int | None = None) -> int:
    """About one clip per 1.5 average clip lengths of video, so clips have room between them."""
    if requested:
        return max(1, min(int(requested), cap))
    average = (min_sec + max_sec) / 2
    return max(1, min(cap, int(duration // (average * 1.5))))