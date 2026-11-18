import logging
import os
import socket
import time
import traceback

from app.config import get_settings
from app.core.db import SessionLocal
from app.worker.queue import (
    Heartbeat, JobCanceled, ProgressReporter, claim_job, fail_job,
    finish_job, release_job, requeue_stale,
)
from app.worker.runner import run_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")


def process(job_id: str, worker_id: str) -> None:
    reporter = ProgressReporter(job_id, worker_id)
    try:
        with Heartbeat(job_id, worker_id):
            reporter.report("starting", 0, "Starting", force=True)
            clips = run_job(job_id, reporter)
        finish_job(job_id, worker_id, clips)
        log.info("job %s done (%d clips)", job_id, len(clips))
    except JobCanceled:
        log.info("job %s was canceled or reassigned; stopping", job_id)
    except KeyboardInterrupt:
        release_job(job_id, worker_id)           # give the job back so another worker can take it
        log.info("job %s returned to the queue", job_id)
        raise
    except Exception:
        log.error("job %s failed:\n%s", job_id, traceback.format_exc())
        fail_job(job_id, worker_id, "Processing failed. Please try again.")


def main() -> None:
    if SessionLocal is None:
        raise SystemExit("DATABASE_URL is not configured")

    settings = get_settings()
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    log.info("worker %s started", worker_id)

    last_reap = 0.0
    try:
        while True:
            if time.monotonic() - last_reap >= 30:
                n = requeue_stale(settings.worker_stale_seconds, settings.worker_max_attempts)
                if n:
                    log.warning("reaper recovered %d stuck job(s)", n)
                last_reap = time.monotonic()

            job_id = claim_job(worker_id)
            if job_id is None:
                time.sleep(settings.worker_poll_seconds)
                continue

            log.info("claimed job %s", job_id)
            process(job_id, worker_id)
    except KeyboardInterrupt:
        log.info("worker stopped")


if __name__ == "__main__":
    main()