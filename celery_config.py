import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np  # noqa: F401
from celery import Celery
from loguru import logger
from main import spec2struct
from config import CACHE_DIR

# --- Celery App Configuration ---
# Create Celery instance
celery_app = Celery("spectrum_processor")

# Configuration
celery_app.conf.update(
    # Broker settings (Redis)
    broker_url="redis://redis:6379/0",
    result_backend="redis://redis:6379/0",
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker settings
    worker_prefetch_multiplier=1,  # Process one task at a time per worker
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,
    # Result settings
    result_expires=3600,  # Results expire after 1 hour
    # Retry settings
    task_default_retry_delay=60,  # Retry after 60 seconds
    task_max_retries=3,
    # Routing (for multiple queues)
    task_routes={
        "spectrum_processor.process_spectrum": {"queue": "spectrum_queue"},
        "spectrum_processor.cleanup_old_results": {"queue": "maintenance_queue"},
    },
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup-old-results": {
            "task": "spectrum_processor.cleanup_old_results",
            "schedule": 3600.0,  # Run every hour
        },
    },
)

# --- Cache Directory ---
CACHE_DIR.mkdir(exist_ok=True)


# --- Helper Function (Unchanged) ---
def short_vector_hash(vector, length=32):
    """Generate a short hash for better readability"""
    full_hash = hashlib.sha256(vector.tobytes()).hexdigest()
    return full_hash[:length]


# --- CORRECTED Celery Task ---
@celery_app.task(bind=True, name="spectrum_processor.process_spectrum")
def process_spectrum(self, job_id: str, request_data: dict):
    """
    Main spectrum processing task. Runs spec2struct and returns the results.
    """
    try:
        start_time = time.time()
        logger.info(
            f"Starting Celery task for job {job_id} (Task ID: {self.request.id})"
        )

        total_gens = request_data.get("gens_ga", 10)
        self.update_state(
            state="PROGRESS",
            meta={
                "status": "Initializing genetic algorithm...",
                "current": 0,
                "total": total_gens,
                "job_id": job_id,
            },
        )

        # Directly call the function
        results = spec2struct(**request_data) or []

        end_time = time.time()
        processing_time = end_time - start_time

        final_payload = {
            "results": results,
            "metadata": {
                "job_id": job_id,
                "processing_time": processing_time,
                "timestamp": time.time(),
                "task_id": self.request.id,
            },
        }

        # Save the result to a file for persistence/backup
        result_file = CACHE_DIR / f"{job_id}.json"
        with result_file.open("w") as f:
            json.dump(final_payload, f)

        logger.info(f"Job {job_id} completed in {processing_time:.2f} seconds")

        # Return the final payload so it's stored in the result backend
        return final_payload

    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}")
        # Let Celery handle the failure state update. The retry mechanism will do this automatically.
        raise self.retry(
            exc=exc,
            countdown=min(60 * (2**self.request.retries), 300),
            max_retries=3,
        )


@celery_app.task(name="spectrum_processor.cleanup_old_results")
def cleanup_old_results():
    """Periodic task to clean up old result files"""
    try:
        current_time = time.time()
        cleaned_count = 0

        for result_file in CACHE_DIR.glob("*.json"):
            try:
                # Check file age (older than 24 hours)
                if current_time - result_file.stat().st_mtime > 86400 * 180:
                    result_file.unlink()
                    cleaned_count += 1
                    logger.info(f"Cleaned up old result file: {result_file}")
            except Exception as e:
                logger.error(f"Error cleaning file {result_file}: {e}")

        logger.info(f"Cleanup task completed. Removed {cleaned_count} old files.")
        return {"cleaned_files": cleaned_count}

    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise
