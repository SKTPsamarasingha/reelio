from datetime import datetime
from typing import Any

from pydantic import (
    BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator,
)
from pydantic.alias_generators import to_camel

from app.config import get_settings
from app.options import (
    AI_PICKS, CAPTION_STYLES, DEFAULTS, FRAMINGS, HOOK_MODES, LAYOUTS, normalize_key, resolve_ai_pick,
)


class CamelModel(BaseModel):
    """camelCase on the wire, snake_case in Python."""
    model_config = ConfigDict(
        alias_generator=to_camel, validate_by_alias=True, validate_by_name=True, from_attributes=True
    )


def _camelize(value: Any) -> Any:
    if isinstance(value, dict):
        return {to_camel(k): _camelize(v) for k, v in value.items()}
    return value


def _one_of(value: str, allowed, name: str) -> str:
    v = normalize_key(value)
    if v not in allowed:
        raise ValueError(f"unknown {name} '{value}'; choose one of {list(allowed)}")
    return v


# ---------- request ----------
class ClipLength(CamelModel):
    label: str = Field("", max_length=40)
    min_sec: int
    max_sec: int

    @model_validator(mode="after")
    def check_range(self):
        s = get_settings()
        if not (s.clip_seconds_floor <= self.min_sec < self.max_sec <= s.clip_seconds_ceiling):
            raise ValueError(
                f"clip length must be between {s.clip_seconds_floor} and {s.clip_seconds_ceiling} "
                "seconds, with minSec below maxSec"
            )
        return self


class JobCreate(CamelModel):
    video_id: str | None = Field(None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    source_type: str
    cloudinary_public_id: str | None = Field(None, max_length=200)
    cloudinary_url: str | None = None          # accepted but never used (see "Things to know")
    source_url: str | None = None              # only when sourceType is "url"
    duration_sec: float | None = Field(None, ge=0)   # informational; the real value comes from Cloudinary

    layout: str = DEFAULTS["layout"]
    framing: str = DEFAULTS["framing"]
    ai_picks: str = DEFAULTS["ai_picks"]
    clip_length: ClipLength = Field(
        default_factory=lambda: ClipLength(min_sec=DEFAULTS["clip_min"], max_sec=DEFAULTS["clip_max"])
    )
    caption_style: str = DEFAULTS["caption_style"]
    hook_mode: str = DEFAULTS["hook_mode"]
    custom_prompt: str = ""
    num_clips: int | None = Field(None, ge=1, le=10)

    @field_validator("source_type", mode="before")
    @classmethod
    def _source_type(cls, v):
        v = str(v).strip().lower()
        if v not in ("upload", "url"):
            raise ValueError("sourceType must be 'upload' or 'url'")
        return v

    @field_validator("layout")
    @classmethod
    def _layout(cls, v):
        v = v.strip()
        if v not in LAYOUTS:
            raise ValueError(f"unknown layout '{v}'; choose one of {list(LAYOUTS)}")
        return v

    @field_validator("framing")
    @classmethod
    def _framing(cls, v):
        return _one_of(v, FRAMINGS, "framing")

    @field_validator("caption_style")
    @classmethod
    def _caption_style(cls, v):
        return _one_of(v, CAPTION_STYLES, "captionStyle")

    @field_validator("hook_mode")
    @classmethod
    def _hook_mode(cls, v):
        return _one_of(v, HOOK_MODES, "hookMode")

    @field_validator("ai_picks")
    @classmethod
    def _ai_picks(cls, v):
        key = resolve_ai_pick(v)
        if key is None:
            raise ValueError(f"unknown aiPicks '{v}'; choose one of {list(AI_PICKS)}")
        return key

    @field_validator("custom_prompt")
    @classmethod
    def _custom_prompt(cls, v):
        limit = get_settings().max_custom_prompt_chars
        v = v.strip()
        if len(v) > limit:
            raise ValueError(f"customPrompt must be at most {limit} characters")
        return v

    @model_validator(mode="after")
    def check_source(self):
        if self.source_type == "upload" and not self.cloudinary_public_id:
            raise ValueError("cloudinaryPublicId is required for uploads")
        if self.source_type == "url" and not (self.source_url or "").startswith(("http://", "https://")):
            raise ValueError("sourceUrl must be a valid http(s) URL")
        return self

    def to_options(self) -> dict:
        """What gets stored on the job and read by the worker."""
        return {
            "video_id": self.video_id,
            "layout": self.layout,
            "framing": self.framing,
            "ai_picks": self.ai_picks,
            "clip_length": {"label": self.clip_length.label,
                            "min_sec": self.clip_length.min_sec, "max_sec": self.clip_length.max_sec},
            "caption_style": self.caption_style,
            "hook_mode": self.hook_mode,
            "custom_prompt": self.custom_prompt,
            "num_clips": self.num_clips,
        }


# ---------- response ----------
class ClipOut(CamelModel):
    id: str
    job_id: str
    rank: int
    title: str
    reasoning: str
    hook_text: str | None = None
    start_sec: float
    end_sec: float
    duration_sec: float
    layout: str
    created_at: datetime
    url: str | None = None            # plays in a <video> tag
    download_url: str | None = None   # forces a download


class ClipPage(CamelModel):
    items: list[ClipOut]
    total: int
    limit: int
    offset: int


class JobSummary(CamelModel):
    id: str
    status: str
    stage: str
    progress: float
    message: str
    source_type: str
    source_url: str | None
    options: dict
    cache_hits: dict
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @field_serializer("options")
    def _options(self, value: dict):
        return _camelize(value)         # tolerant of old jobs with a different shape


class JobDetail(JobSummary):
    clips: list[ClipOut] = []