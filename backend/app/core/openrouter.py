# app/core/openrouter.py
from openai import OpenAI
from app.config import get_settings


def get_openrouter_client() -> OpenAI:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured in environment variables.")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )