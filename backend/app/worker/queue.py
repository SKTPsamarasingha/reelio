import threading
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.core.db import SessionLocal
from app.models import Clip


class JobCanceled(Exception):
    """Raised when the job was canceled, or this worker no longer owns it."""


def claim_job(worker_id: str) -> str | None:
    """Atomically take the oldest queued job. Safe with several workers running."""
    with SessionLocal() as db:
        row = db.execute(
            text("""
                UPDATE jobs
                SET status = 'RUNNING',
                    worker_id = :wid,
                    locked_at = now(),
                    started_at = COALESCE(started_at, now()),
                    attempts = attempts + 1,
                    stage = 'starting',
                    progress = 0
                WHERE id = (
                    SELECT id FROM jobs
                    WHERE status = 'QUEUED'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING id
            """),
            {"wid": worker_id},
        ).first()
        db.commit()
        return row[0] if row else None


def touch(job_id: str, worker_id: str) -> bool:
    """Heartbeat. Returns False if the job is no longer ours."""
    with SessionLocal() as db:
        res = db.execute(
            text("""UPDATE jobs SET locked_at = now()
                    WHERE id = :id AND status = 'RUNNING' AND worker_id = :wid"""),
            {"id": job_id, "wid": worker_id},
        )
        db.commit()
        return res.rowcount == 1


def write_progress(job_id: str, worker_id: str, stage: str, progress: float, message: str) -> None:
    with SessionLocal() as db:
        res = db.execute(
            text("""UPDATE jobs
                    SET stage = :stage, progress = :progress, message = :message, locked_at = now()
                    WHERE id = :id AND status = 'RUNNING' AND worker_id = :wid"""),
            {"id": job_id, "wid": worker_id, "stage": stage,
             "progress": round(progress, 1), "message": message},
        )
        db.commit()
        if res.rowcount != 1:
            raise JobCanceled(job_id)


def finish_job(job_id: str, worker_id: str, clips: list[dict]) -> None:
    """Mark DONE and save the clips in one transaction."""
    with SessionLocal() as db:
        res = db.execute(
            text("""UPDATE jobs
                    SET status = 'DONE', progress = 100, stage = 'done',
                        message = 'Finished', finished_at = now(), error = NULL
                    WHERE id = :id AND status = 'RUNNING' AND worker_id = :wid"""),
            {"id": job_id, "wid": worker_id},
        )
        if res.rowcount != 1:
            db.rollback()
            raise JobCanceled(job_id)
        for c in clips:
            db.add(Clip(job_id=job_id, **c))
        db.commit()


def fail_job(job_id: str, worker_id: str, error: str) -> None:
    with SessionLocal() as db:
        db.execute(
            text("""UPDATE jobs
                    SET status = 'FAILED', error = :error, finished_at = now()
                    WHERE id = :id AND status = 'RUNNING' AND worker_id = :wid"""),
            {"id": job_id, "wid": worker_id, "error": error[:500]},
        )
        db.commit()


def release_job(job_id: str, worker_id: str) -> None:
    """Put a running job back in the queue (used when the worker is shut down)."""
    with SessionLocal() as db:
        db.execute(
            text("""UPDATE jobs
                    SET status = 'QUEUED', worker_id = NULL, locked_at = NULL,
                        attempts = GREATEST(attempts - 1, 0)
                    WHERE id = :id AND status = 'RUNNING' AND worker_id = :wid"""),
            {"id": job_id, "wid": worker_id},
        )
        db.commit()


def requeue_stale(stale_seconds: int, max_attempts: int) -> int:
    """Reaper: recover jobs whose worker disappeared. Returns how many were touched."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
    with SessionLocal() as db:
        res = db.execute(
            text("""
                UPDATE jobs
                SET status = CASE WHEN attempts >= :max THEN 'FAILED' ELSE 'QUEUED' END,
                    error = CASE WHEN attempts >= :max
                                 THEN 'The worker stopped responding.' ELSE error END,
                    finished_at = CASE WHEN attempts >= :max THEN now() ELSE finished_at END,
                    worker_id = NULL,
                    locked_at = NULL
                WHERE status = 'RUNNING' AND locked_at < :cutoff
            """),
            {"max": max_attempts, "cutoff": cutoff},
        )
        db.commit()
        return res.rowcount

def reassigned(job_id: str, worker_id: str) -> bool:
    """True if the job has been queued or taken by another worker since this worker started it."""
    with SessionLocal() as db:
        row = db.execute(text("SELECT status, worker_id FROM jobs WHERE id = :id"), {"id": job_id}).first()
    return bool(row and row[0] in ("QUEUED", "RUNNING") and row[1] != worker_id)

class Heartbeat:
    """Context manager: refreshes locked_at every few seconds while a job runs."""

    def __init__(self, job_id: str, worker_id: str, interval: float = 15.0):
        self.job_id, self.worker_id, self.interval = job_id, worker_id, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                touch(self.job_id, self.worker_id)
            except Exception:
                pass            # a failed heartbeat must never crash the job

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


class ProgressReporter:
    """Writes progress to the database at most once per second (stage changes always go through)."""

    def __init__(self, job_id: str, worker_id: str, min_interval: float = 1.0):
        self.job_id, self.worker_id, self.min_interval = job_id, worker_id, min_interval
        self._last_write = 0.0
        self._last_stage = ""

    def report(self, stage: str, progress: float, message: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if force or stage != self._last_stage or now - self._last_write >= self.min_interval:
            write_progress(self.job_id, self.worker_id, stage, progress, message)
            self._last_write, self._last_stage = now, stage