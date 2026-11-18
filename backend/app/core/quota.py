from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.errors import ApiError
from app.models import Job, JobStatus, User


def check_quota(db: Session, user: User) -> None:
    """Single place for limits. Payments/credits plug in here later."""
    s = get_settings()
    active = db.scalar(
        select(func.count(Job.id)).where(
            Job.user_id == user.id, Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
        )
    )
    if active >= s.max_active_jobs_per_user:
        raise ApiError(429, "too_many_active_jobs",
                       f"You can have at most {s.max_active_jobs_per_user} jobs running at once.")

# from sqlalchemy import func, select
# from sqlalchemy.orm import Session
#
# from app.config import get_settings
# from app.core.errors import ApiError
# from app.models import Job, JobStatus, User
# from app.schemas import JobCreate
#
#
# def check_quota(db: Session, user: User, payload: JobCreate) -> None:
#     """Single place for limits. Payments/credits plug in here later."""
#     s = get_settings()
#
#     if payload.options.num_clips > s.max_clips_per_job:
#         raise ApiError(422, "too_many_clips", f"At most {s.max_clips_per_job} clips per job.")
#
#     active = db.scalar(
#         select(func.count(Job.id)).where(
#             Job.user_id == user.id,
#             Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
#         )
#     )
#     if active >= s.max_active_jobs_per_user:
#         raise ApiError(
#             429, "too_many_active_jobs",
#             f"You can have at most {s.max_active_jobs_per_user} jobs running at once.",
#         )