import pytest
from pydantic import ValidationError

from app.schemas import JobCreate


def test_valid_url_job():
    job = JobCreate(source_type="URL", source_url="https://example.com/v.mp4")
    assert job.options.num_clips == 3
    assert job.options.platforms == ["instagram_reels"]


def test_url_job_needs_http_url():
    with pytest.raises(ValidationError):
        JobCreate(source_type="URL", source_url="file:///etc/passwd")


def test_upload_needs_key():
    with pytest.raises(ValidationError):
        JobCreate(source_type="UPLOAD")


def test_unknown_platform_rejected():
    with pytest.raises(ValidationError):
        JobCreate(source_type="URL", source_url="https://a.com/x",
                  options={"platforms": ["myspace"]})