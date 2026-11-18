import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy import delete, select, update

from app.models import Clip, Job, JobStatus, User
from app.config import get_settings
from app.core.auth import get_current_user
from app.core.db import get_db
from app.core.errors import ApiError
from app.core.quota import check_quota
from app.core.storage import clip_urls, delete_job_assets, inspect_video, is_video_public_id
from app.core.urlsafe import is_public_url
from app.models import Job, JobStatus, User
from app.schemas import JobCreate, JobDetail, JobSummary

log = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


def _with_urls(job: Job) -> JobDetail:
    detail = JobDetail.model_validate(job)
    for out, row in zip(detail.clips, job.clips):
        if row.storage_key:
            out.url, out.download_url = clip_urls(row.storage_key)
    return detail


def _get_owned_job(db: Session, user: User, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:      # 404, not 403, so ids can't be probed
        raise ApiError(404, "job_not_found", "Job not found.")
    return job


def _upload_source(public_id: str, user: User) -> dict:
    s = get_settings()
    public_id = (public_id or "").strip()
    if not is_video_public_id(public_id) or f"/videos/{user.id}/" not in f"/{public_id}":
        raise ApiError(422, "invalid_upload", "That upload reference isn't valid.")

    try:
        info = inspect_video(public_id)
    except Exception:
        log.exception("could not inspect upload %s", public_id)
        raise ApiError(503, "storage_unavailable", "Couldn't check your upload. Please try again.")
    if info is None:
        raise ApiError(404, "upload_not_found", "We couldn't find that upload. Please upload it again.")

    if info["duration"] and info["duration"] > s.max_video_seconds:
        raise ApiError(422, "video_too_long", f"Videos can be at most {s.max_video_seconds // 60} minutes long.")
    if info["bytes"] > s.max_upload_mb * 1024 * 1024:
        raise ApiError(422, "video_too_large", f"Videos can be at most {s.max_upload_mb} MB.")

    return {"source_type": "UPLOAD", "source_key": public_id, "source_url": None,
            "video_key": f"cld:{public_id}:v{info['version']}"}


def _url_source(url: str | None) -> dict:
    if not get_settings().allow_url_sources:
        raise ApiError(422, "url_not_supported", "Please upload your video file.")
    if not is_public_url(url):
        raise ApiError(422, "invalid_url", "That link can't be used.")
    return {"source_type": "URL", "source_url": url, "source_key": None, "video_key": None}


@router.post("", response_model=JobSummary, status_code=201)
def create_job(payload: JobCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if payload.source_type == "upload":
        source = _upload_source(payload.cloudinary_public_id, user)
    else:
        source = _url_source(payload.source_url)

    check_quota(db, user)

    job = Job(user_id=user.id, status=JobStatus.QUEUED, stage="queued", options=payload.to_options(), **source)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@router.get("", response_model=list[JobSummary])
def list_jobs(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
              db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    stmt = (select(Job).where(Job.user_id == user.id)
            .order_by(Job.created_at.desc()).limit(limit).offset(offset))
    return db.scalars(stmt).all()


@router.get("/{job_id}", response_model=JobDetail)
def get_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return _with_urls(_get_owned_job(db, user, job_id))


@router.post("/{job_id}/cancel", response_model=JobSummary)
def cancel_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = _get_owned_job(db, user, job_id)
    if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise ApiError(409, "not_cancelable", "Only queued or running jobs can be canceled.")
    job.status = JobStatus.CANCELED
    job.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)
    return job

@router.post("/{job_id}/retry", response_model=JobSummary)
def retry_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = _get_owned_job(db, user, job_id)
    if job.status not in (JobStatus.FAILED, JobStatus.CANCELED):
        raise ApiError(409, "not_retryable", "Only failed or canceled jobs can be retried.")

    # The original source must still be usable, and the user must have room for another active job.
    if job.source_type == "UPLOAD":
        _upload_source(job.source_key, user)
    else:
        _url_source(job.source_url)
    check_quota(db, user)

    delete_job_assets(job.id)       # leftovers from the earlier attempt (best effort)

    # One atomic step: if two requests race, only the first one matches.
    res = db.execute(
        update(Job)
        .where(Job.id == job.id, Job.status.in_([JobStatus.FAILED, JobStatus.CANCELED]))
        .values(
            status=JobStatus.QUEUED, stage="queued", progress=0.0, message="", error=None,
            attempts=0, worker_id=None, locked_at=None, started_at=None, finished_at=None,
            cache_hits={},
        )
    )
    if res.rowcount != 1:
        db.rollback()
        raise ApiError(409, "not_retryable", "This job was already retried.")

    db.execute(delete(Clip).where(Clip.job_id == job.id))
    db.commit()
    db.refresh(job)
    return job

@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = _get_owned_job(db, user, job_id)
    if job.status == JobStatus.RUNNING:
        raise ApiError(409, "job_running", "Cancel the job before deleting it.")
    delete_job_assets(job.id)       # clips and sheets only; the uploaded source is not touched
    db.delete(job)
    db.commit()


