from fastapi import APIRouter, Depends

from app.core.auth import get_current_user
from app.core.errors import ApiError
from app.core.storage import is_configured, sign_upload
from app.models import User

router = APIRouter(tags=["uploads"])


@router.post("/uploads/sign")
def sign(user: User = Depends(get_current_user)):
    if not is_configured():
        raise ApiError(503, "storage_unavailable", "Uploads are not available right now.")
    return sign_upload(user.id)