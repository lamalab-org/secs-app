# SECS: Structure Elucidation from Chemical Spectra

## API Reference

The `app.py` file defines a FastAPI application that provides an API for submitting and managing spectrum processing jobs. It uses Celery for asynchronous task processing and Redis for storing job state. Completed results are persisted as JSON files in the cache directory (`<job_id>.json`, stamped with `"completed": true` by the worker); these files are the source of truth for `/submit` deduplication and `/jobs/{job_id}/result`. While a job runs, the genetic algorithm additionally writes an in-progress snapshot to `<job_id>.partial.json`, which feeds live progress into `/jobs/{job_id}/status`.

### Endpoints

#### Submit a Job

- **POST** `/submit`
- Submits a new spectrum processing job to the queue. The `job_id` is a hash of the spectrum's `y` values, so byte-identical spectra always map to the same job. Deduplication happens in two stages: if a **completed** result for that spectrum is already cached, no new job is queued; if a task for that spectrum is **still running**, the existing job is returned instead of enqueuing a duplicate.
- **Request Body**:
  - `mf` (str): Molecular formula.
  - `spectrum` (dict): A dictionary containing the spectrum data (e.g., `{"y": [y_values]}`).
  - `model` (str, optional): The model to use. Defaults to `"residual"`.
  - `configs_path` (str, optional): Path to configs. Defaults to `"configs"`.
  - `ga_ir_exp` (str | None, optional): IR experiment config.
  - `ga_cnmr_exp` (str | None, optional): CNMR experiment config.
  - `ga_hnmr_exp` (str | None, optional): HNMR experiment config. Defaults to `"test/hnmr_augment_finetune_residual"`.
  - `ga_hsqc_exp` (str | None, optional): HSQC experiment config.
  - `seed` (int, optional): Random seed. Defaults to `42`.
  - `init_pop_ga` (int, optional): Initial population size for GA. Defaults to `512`.
  - `frac_graph_ga_mutate` (float, optional): Fraction of graph to mutate in GA. Defaults to `0.3`.
  - `gens_ga` (int, optional): Number of generations for GA. Defaults to `10`.
  - `offspring_ga` (int, optional): Number of offspring for GA. Defaults to `1024`.
  - `pop_ga` (int, optional): Population size for GA. Defaults to `512`.
  - `initial_environment` (dict | None, optional): Initial environment settings.
- **Returns**:
  - New job: `{"job_id", "task_id", "status": "submitted", "message"}`.
  - Job already running: `{"job_id", "task_id", "status": "already_submitted", "message"}` — poll `/jobs/{job_id}/status` as usual.
  - Cached result: `{"job_id", "status": "cached", "message"}` — note there is **no** `task_id`. Clients should check `status` and, when it is `"cached"`, skip polling and call `/jobs/{job_id}/result` directly.
- **Caveat**: because the `job_id` hashes only the spectrum's `y` values, submitting the same spectrum with different parameters (`mf`, `model`, GA settings) deduplicates against the earlier run and does **not** start a new one. Clear the cache entry first if a re-run with new parameters is intended.

#### Get Job Status

- **GET** `/jobs/{job_id}/status`
- Retrieves the current status of a specific job via its Celery task. While the job runs, the response is enriched from the GA's partial snapshot with:
  - `results`: the current top-scoring candidates,
  - `stage`: human-readable phase (`"initial population"`, `"generation N"`),
  - `generation`: current GA generation as an integer,
  - `n_evaluated`: total number of candidate molecules scored so far.
- **Path Parameter**:
  - `job_id` (str): The ID of the job.
- **Returns**: A JSON object with the job's `status` (lower-cased Celery state, e.g. `pending`, `progress`, `success`, `failure`), plus the progress fields above while running, `{"completed": true}` on success, or error details (`error`, `traceback`) on failure.
- **Note**: Returns `404` when no `job_id -> task_id` mapping exists in Redis — e.g. for cached results (no task was queued) or after the mapping has expired (24 h). In that case the result may still be available via `/jobs/{job_id}/result`.

#### Get Job Result

- **GET** `/jobs/{job_id}/result`
- Fetches the result of a completed job. The stored cache file is the source of truth, but only when it carries the worker's `"completed": true` stamp: a completed result is served regardless of Celery task state (and therefore remains retrievable after the Celery result and Redis mapping expire), while an in-progress or truncated file is never served as final.
- **Path Parameter**:
  - `job_id` (str): The ID of the job.
- **Returns**:
  - `200` with the result payload: `{"completed": true, "query": <submitted request>, "results": [...], "metadata": {...}}`.
  - `400` if the job is known but not yet completed, with the current task status in `detail`.
  - `410` if job files exist but the task is no longer tracked (the run died mid-way or the mapping expired before completion).
  - `404` if no result is stored and the job is unknown.

#### Cancel a Job

- **DELETE** `/jobs/{job_id}`
- Cancels a pending or running job. Also removes the job's Redis mapping and its in-progress snapshot, so a subsequent `/submit` of the same spectrum starts a clean run.
- **Path Parameter**:
  - `job_id` (str): The ID of the job to cancel.
- **Returns**: A JSON object confirming the cancellation with `job_id`, `task_id`, and `status`.

#### Get Queue Statistics

- **GET** `/queue/stats`
- Provides statistics about the Celery queue, including the number of active, scheduled, and reserved tasks.
- **Returns**: A JSON object with queue statistics.

#### Get Worker Information

- **GET** `/workers`
- Retrieves information about the active Celery workers.
- **Returns**: A JSON object containing worker stats.

#### Clear Cache

- **DELETE** `/cache`
- Deletes all files from the cache directory, including completed results and in-progress snapshots. Subsequent `/submit` calls for previously cached spectra will queue new runs.
- **Returns**: A JSON object confirming the cache has been cleared.