import json

import numpy as np
import redis
from celery.result import AsyncResult
from celery_config import celery_app, short_vector_hash
from config import CACHE_DIR
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Spectrum Processing API with Celery", version="2.0.0")
redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)


class GenerateRequest(BaseModel):
    mf: str
    spectrum: dict[str, list[float]]
    model: str = "residual"
    configs_path: str = "configs"
    ga_ir_exp: str | None = None
    ga_cnmr_exp: str | None = None
    ga_hnmr_exp: str | None = "test/hnmr_augment_finetune_residual"
    ga_hsqc_exp: str | None = None
    seed: int = 42
    init_pop_ga: int = 512
    frac_graph_ga_mutate: float = 0.3
    gens_ga: int = 10
    offspring_ga: int = 1024
    pop_ga: int = 512
    initial_environment: dict[str, str] | None = None


@app.post("/submit")
async def submit_spectrum_job(request: GenerateRequest):
    """Submit a spectrum processing job to the Celery queue"""
    try:
        spectrum_as_array = np.array(request.spectrum["y"])
        job_id = short_vector_hash(spectrum_as_array)

        request_data = request.model_dump()
        request_data["spectra_hash"] = job_id

        task = celery_app.send_task(
            "spectrum_processor.process_spectrum",
            args=[job_id, request_data],
            queue="spectrum_queue",
        )

        # Store mapping in Redis
        redis_client.set(job_id, task.id, ex=86400)

        return {
            "job_id": job_id,
            "task_id": task.id,
            "status": "submitted",
            "message": "Job submitted to queue",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {e!s}")


def get_task_id_from_job_id(job_id: str) -> str:
    """Helper function to get task_id from Redis."""
    task_id = redis_client.get(job_id)
    if not task_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return task_id


@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str):
    """Get the current status of a job"""
    try:
        task_id = get_task_id_from_job_id(job_id)
        task_result = AsyncResult(task_id, app=celery_app)

        if not task_result:
            raise HTTPException(
                status_code=500, detail="Could not retrieve task result."
            )

        response = {
            "job_id": job_id,
            "task_id": task_id,
            "status": task_result.state.lower(),
        }

        if task_result.state == "PROGRESS":
            if isinstance(task_result.info, dict):
                response.update(task_result.info)
        elif task_result.state == "SUCCESS":
            response.update({"completed": True, "result": task_result.result})
        elif task_result.state == "FAILURE":
            try:
                error_info = {
                    "error": str(task_result.info),
                    "traceback": task_result.traceback,
                }
            except Exception:
                error_info = {
                    "error": "Could not retrieve error details.",
                    "traceback": None,
                }
            response.update(error_info)

        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e!s}"
        )  # noqa: B904


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """Get the result of a completed job"""
    # --- MODIFIED: Get task_id from Redis ---
    task_id = get_task_id_from_job_id(job_id)
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state != "SUCCESS":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {task_result.state}",
        )

    result_file = CACHE_DIR / f"{job_id}_cache.json"
    # try:
    with result_file.open("r") as f:
        data = json.load(f)
    return data
    # except FileNotFoundError:
    #     raise HTTPException(status_code=404, detail="Result file not found")  # noqa: B904
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error loading results: {e!s}")  # noqa: B904


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    """Cancel a pending or running job"""
    task_id = get_task_id_from_job_id(job_id)
    celery_app.control.revoke(task_id, terminate=True)

    # Clean up the mapping from Redis
    redis_client.delete(job_id)

    return {"job_id": job_id, "task_id": task_id, "status": "cancelled"}


@app.get("/queue/stats")
def get_queue_stats():
    """Get queue statistics"""
    try:
        # Get active tasks
        active_tasks = celery_app.control.inspect().active()

        # Get scheduled tasks
        scheduled_tasks = celery_app.control.inspect().scheduled()

        # Get reserved tasks
        reserved_tasks = celery_app.control.inspect().reserved()

        # Count tasks
        active_count = sum(len(tasks) for tasks in (active_tasks or {}).values())
        scheduled_count = sum(len(tasks) for tasks in (scheduled_tasks or {}).values())
        reserved_count = sum(len(tasks) for tasks in (reserved_tasks or {}).values())

        return {
            "active_tasks": active_count,
            "scheduled_tasks": scheduled_count,
            "reserved_tasks": reserved_count,
            "total_submitted_jobs": redis_client.dbsize(),
            "workers": list((active_tasks or {}).keys()) if active_tasks else [],
        }

    except Exception as e:
        return {
            "error": f"Could not fetch queue stats: {e!s}",
            "total_submitted_jobs": redis_client.dbsize(),
        }


@app.get("/workers")
def get_worker_info():
    """Get information about active workers"""
    try:
        stats = celery_app.control.inspect().stats()
        return {"workers": stats or {}}
    except Exception as e:
        return {"error": f"Could not fetch worker info: {e!s}"}


@app.delete("/cache")
def clear_cache():
    """Deletes all files in the cache directory."""
    try:
        deleted_count = 0
        for item in CACHE_DIR.glob("*"):
            if item.is_file():
                item.unlink()
                deleted_count += 1
        return {
            "status": "success",
            "message": f"Cleared {deleted_count} files from the cache.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e!s}")
