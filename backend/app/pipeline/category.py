"""Video category and the editing hint that goes with it."""
import re

YT_MAP = {
    "Film & Animation": "movie_scene", "Gaming": "gaming", "Education": "tutorial", "Howto & Style": "tutorial",
    "Comedy": "comedy", "News & Politics": "news_politics", "Sports": "sports", "Music": "music",
    "Science & Technology": "tech_explainer", "People & Blogs": "vlog_talk", "Entertainment": "entertainment",
}

KEYWORDS = {
    "movie_scene": ["movie clip", "official clip", "scene", "trailer", "film", "movie"],
    "podcast_interview": ["podcast", "interview", "episode", "conversation", "q&a"],
    "debate": ["debate", "vs", "argues", "heated", "hearing", "court"],
    "comedy": ["stand-up", "standup", "funny", "comedy", "sketch", "roast"],
    "tutorial": ["how to", "tutorial", "guide", "learn", "tips", "explained"],
    "gaming": ["gameplay", "walkthrough", "let's play", "speedrun"],
    "news_politics": ["breaking", "news", "election", "senate", "press conference"],
    "sports": ["highlights", "match", "goal", "nba", "nfl", "ufc"],
}

# What makes a clip work in each niche. Goes into the payload for the model.
CLIP_HINTS = {
    "movie_scene": "Look for dramatic reveals, standout action beats, memorable one-liners and emotional peaks.",
    "podcast_interview": "Look for strong opinions, surprising claims, stories with a payoff and quotable lines.",
    "debate": "Look for direct clashes, interruptions, rebuttals and decisive moments.",
    "comedy": "Look for setup and punchline pairs. Include the full setup and the audience reaction.",
    "tutorial": "Look for a single complete tip with a clear result. Avoid cutting mid-step.",
    "gaming": "Look for clutch plays, big reactions and unexpected events.",
    "news_politics": "Look for key statements, confrontations and newsworthy announcements.",
    "sports": "Look for decisive plays and crowd or commentator peaks.",
    "tech_explainer": "Look for a clear self-contained explanation or a surprising insight.",
    "vlog_talk": "Look for personal stories, emotional moments and strong opinions.",
    "entertainment": "Look for high-energy, surprising or funny moments.",
    "music": "Look for the hook, the drop, a standout vocal or instrumental moment, and crowd energy.",
    "general": "Look for self-contained moments with a clear beginning and end.",
}


def _count(text: str, keyword: str) -> int:
    """Whole-word matches only, so 'vs' doesn't match inside 'canvas'."""
    return len(re.findall(rf"(?<!\w){re.escape(keyword)}(?!\w)", text))


def detect_category(meta: dict, segments=None) -> str:
    """From title/description/tags, then YouTube's own category. With neither, 'general':
    the model works out the video type from the transcript and frames instead of us guessing."""
    text = " ".join([meta.get("title") or "", meta.get("context") or "", " ".join(meta.get("tags") or [])]).lower()
    scores = {c: sum(_count(text, k) for k in kws) for c, kws in KEYWORDS.items()}
    best, hits = max(scores.items(), key=lambda kv: kv[1])
    if hits > 0:
        return best
    for c in meta.get("categories") or []:
        if c in YT_MAP:
            return YT_MAP[c]
    return "general"


def clip_hint(category: str) -> str:
    return CLIP_HINTS.get(category, CLIP_HINTS["general"])
