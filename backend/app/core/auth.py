from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.db import get_db
from app.core.errors import ApiError
from app.models import User


def get_current_user(
    x_dev_user: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    settings = get_settings()
    if settings.auth_mode != "dev":
        # Clerk JWT verification is added in a later step.
        raise ApiError(501, "auth_not_configured", "Authentication is not configured.")

    clerk_id = x_dev_user or "dev-user"
    user = db.scalar(select(User).where(User.clerk_id == clerk_id))
    if user is None:
        user = User(clerk_id=clerk_id)
        db.add(user)
        try:
            db.commit()
        except IntegrityError:          # two requests created it at the same moment
            db.rollback()
            user = db.scalar(select(User).where(User.clerk_id == clerk_id))
    return user