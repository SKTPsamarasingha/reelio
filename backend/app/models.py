import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


def new_id() -> str:
    return uuid.uuid4().hex


class JobStatus:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    clerk_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    credits: Mapped[int] = mapped_column(Integer, default=0)   # unused until payments
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    jobs: Mapped[list["Job"]] = relationship(back_populates="user")


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_status_created", "status", "created_at"),
        Index("ix_jobs_user_created", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED)
    stage: Mapped[str] = mapped_column(String(64), default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")

    source_type: Mapped[str] = mapped_column(String(16))              # UPLOAD | URL
    source_url: Mapped[str | None] = mapped_column(Text)
    source_key: Mapped[str | None] = mapped_column(Text)              # R2 key for uploads
    video_key: Mapped[str | None] = mapped_column(String(255), index=True)   # cache identity

    options: Mapped[dict] = mapped_column(JSONB, default=dict)
    cache_hits: Mapped[dict] = mapped_column(JSONB, default=dict)

    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    worker_id: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="jobs")
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Clip.rank"
    )


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)

    rank: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(Text)
    reasoning: Mapped[str] = mapped_column(Text, default="")
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    duration_sec: Mapped[float] = mapped_column(Float)
    layout: Mapped[str] = mapped_column(String(16))                 # was: platform
    hook_text: Mapped[str | None] = mapped_column(Text)
    storage_key: Mapped[str | None] = mapped_column(Text)    # R2 key; download URLs are made on demand

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="clips")


class Artifact(Base):
    """Index of cached stage results (transcript, diarization, signals, analysis...)."""

    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint(
            "video_key", "stage", "params_hash", "pipeline_version", "scope", "owner_id",
            name="uq_artifact_identity",
        ),
        Index("ix_artifacts_last_used", "last_used_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    video_key: Mapped[str] = mapped_column(String(255))
    stage: Mapped[str] = mapped_column(String(32))
    params_hash: Mapped[str] = mapped_column(String(64))
    pipeline_version: Mapped[str] = mapped_column(String(16))
    scope: Mapped[str] = mapped_column(String(8))              # GLOBAL | USER
    owner_id: Mapped[str] = mapped_column(String(32), default="")   # "" for GLOBAL, user id for USER

    storage_key: Mapped[str] = mapped_column(Text)             # where the blob lives in R2
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())