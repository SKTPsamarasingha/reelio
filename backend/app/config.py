from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]   # .../backend

class Settings(BaseSettings):
    """All configuration comes from environment variables or backend/.env."""

    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    database_url: str = ""

    redis_url: str = ""
    video_cache_days: int = 7

    # r2_account_id: str = ""
    # r2_access_key_id: str = ""
    # r2_secret_access_key: str = ""
    # r2_bucket: str = ""

    data_dir: Path = BACKEND_DIR / "data"
    cloudinary_url: str = ""
    cloudinary_folder: str = "reelio"
    upload_sheets: bool = False        # sheets are only used locally by Gemini; the frontend never needs them
    max_video_seconds: int = 3600
    video_max_height: int = 1080  # Default max height limit for yt-dlp
    # limits
    max_upload_mb: int = 500            # check your Cloudinary plan's video size limit
    clip_seconds_floor: int = 5
    clip_seconds_ceiling: int = 180
    max_custom_prompt_chars: int = 500
    allow_url_sources: bool = True     # YouTube-style links, for local testing only

    # pipeline (these used to be per-job options)
    whisper_model: str = "small"
    whisper_language: str = "en"
    diarize: bool = False
    caption_font: str = "Montserrat ExtraBold"

    # clerk_jwks_url: str = ""
    # clerk_issuer: str = ""

    gemini_api_key: str = ""
    gemini_model: str = ""

    # openrouter_api_key: str = ""
    # openrouter_default_model: str =""

    pipeline_version: str = "1"
    web_origin: str = "http://localhost:3000"

    environment: str = "development"      # set to "production" when deployed
    auth_mode: str = "dev"                # "dev" now, "clerk" later

    max_active_jobs_per_user: int = 3
    max_clips_per_job: int = 5

    worker_poll_seconds: float = 2.0
    worker_stale_seconds: int = 300        # job is considered stuck after this long without a heartbeat
    worker_max_attempts: int = 2
    fake_step_seconds: float = 0.3         # speed of the fake pipeline (testing only)

@lru_cache
def get_settings() -> Settings:
    return Settings()


