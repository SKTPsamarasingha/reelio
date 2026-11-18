from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import clips, jobs,uploads
from app.config import get_settings
from app.core.cache import get_redis
from app.core.db import engine
from app.core.errors import ApiError

settings = get_settings()

if settings.environment == "production" and settings.auth_mode == "dev":
    raise RuntimeError("AUTH_MODE=dev is not allowed when ENVIRONMENT=production")

app = FastAPI(title="Reelio API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.web_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError):
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    where = ".".join(str(p) for p in first["loc"] if p != "body")
    return _error(422, "invalid_request", f"{where}: {first['msg']}" if where else first["msg"])


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, "http_error", str(exc.detail))


app.include_router(jobs.router)
app.include_router(clips.router)
app.include_router(uploads.router)

@app.get("/health")
def health():
    checks = {}

    # Database: required
    if engine is None:
        checks["database"] = "not configured"
    else:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception:
            checks["database"] = "error"

    # Redis: optional. The app works without it, so a failure is "degraded", never "down".
    if not settings.redis_url:
        checks["redis"] = "not configured"
    else:
        checks["redis"] = "ok" if get_redis() else "unavailable"

    db_ok = checks["database"] in ("ok", "not configured")
    redis_ok = checks["redis"] in ("ok", "not configured")

    return {
        "status": "ok" if db_ok and redis_ok else "degraded",
        "pipeline_version": settings.pipeline_version,
        "checks": checks,
    }