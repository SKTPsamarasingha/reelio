from fastapi import APIRouter, Depends, Query
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.db import get_db
from app.core.errors import ApiError
from app.core.storage import clip_urls, delete_clip_asset
from app.models import Clip, Job, User
from app.schemas import ClipOut, ClipPage

router = APIRouter(tags=["clips"])

def _to_out(clip: Clip) -> ClipOut:
    out = ClipOut.model_validate(clip)
    if clip.storage_key:
        out.url, out.download_url = clip_urls(clip.storage_key)
    return out


def _owned_clip(db: Session, user: User, clip_id: str) -> Clip:
    clip = db.scalar(
        select(Clip)
        .join(Job, Clip.job_id == Job.id)
        .where(Clip.id == clip_id, Job.user_id == user.id)
    )
    # 404 (not 403) for other people's clips, so ids can't be probed.
    if clip is None:
        raise ApiError(404, "clip_not_found", "Clip not found.")
    return clip


@router.get("/clips", response_model=ClipPage)
def list_clips(
    layout: str | None = Query(None, description="e.g. 9:16"),
    job_id: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    filters = [Job.user_id == user.id]
    if layout:
        filters.append(Clip.layout == layout)
    if job_id:
        filters.append(Clip.job_id == job_id)

    total = db.scalar(
        select(func.count(Clip.id)).join(Job, Clip.job_id == Job.id).where(*filters)
    )
    rows = db.scalars(
        select(Clip)
        .join(Job, Clip.job_id == Job.id)
        .where(*filters)
        .order_by(Clip.created_at.desc(), Clip.rank)
        .limit(limit)
        .offset(offset)
    ).all()
    return ClipPage(items=[_to_out(c) for c in rows], total=total or 0, limit=limit, offset=offset)


@router.get("/jobs/{job_id}/clips", response_model=list[ClipOut])
def list_job_clips(
    job_id: str,
    platform: str | None = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise ApiError(404, "job_not_found", "Job not found.")

    stmt = select(Clip).where(Clip.job_id == job_id).order_by(Clip.rank)
    if platform:
        stmt = stmt.where(Clip.platform == platform)
    return [_to_out(c) for c in db.scalars(stmt).all()]


@router.get("/clips/{clip_id}", response_model=ClipOut)
def get_clip(
    clip_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return _to_out(_owned_clip(db, user, clip_id))


@router.get("/clips/{clip_id}/download")
def download_clip(
    clip_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    clip = _owned_clip(db, user, clip_id)
    out = _to_out(clip)
    if not out.download_url:
        raise ApiError(404, "clip_file_missing", "This clip has no file.")
    return RedirectResponse(out.download_url, status_code=307)


@router.delete("/clips/{clip_id}", status_code=204)
def delete_clip(
    clip_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    clip = _owned_clip(db, user, clip_id)
    delete_clip_asset(clip.storage_key)
    db.delete(clip)
    db.commit()