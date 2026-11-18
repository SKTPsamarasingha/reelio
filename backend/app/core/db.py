from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


def normalize_url(url: str) -> str:
    """Make any Postgres URL use the psycopg 3 driver."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


class Base(DeclarativeBase):
    pass


_settings = get_settings()
DATABASE_URL = normalize_url(_settings.database_url)

# Engine is created only when a URL is set, so the app still starts without a database.
engine = (
    create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=3, max_overflow=2)
    if DATABASE_URL
    else None
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False) if engine else None


def get_db():
    """FastAPI dependency: one session per request."""
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL is not configured")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

