# SECS: Structure Elucidation from Chemical Spectra

## API Reference

The `app.py` file defines a FastAPI application that provides an API for submitting and managing spectrum processing jobs. It uses Celery for asynchronous task processing and Redis for storing job state.

### Endpoints

#### Submit a Job

- **POST** `/submit`
- Submits a new spectrum processing job to the queue.
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
- **Returns**: A JSON object with `job_id`, `task_id`, `status`, and a confirmation `message`.

#### Get Job Status

- **GET** `/jobs/{job_id}/status`
- Retrieves the current status of a specific job.
- **Path Parameter**:
  - `job_id` (str): The ID of the job.
- **Returns**: A JSON object with the job's `job_id`, `task_id`, and `status`. If the job is in progress, it may contain progress information. If it has failed, it will contain error details.

#### Get Job Result

- **GET** `/jobs/{job_id}/result`
- Fetches the result of a completed job. The result is read from a cache file.
- **Path Parameter**:
  - `job_id` (str): The ID of the job.
- **Returns**: The job's result data from the cache file as JSON.

#### Cancel a Job

- **DELETE** `/jobs/{job_id}`
- Cancels a pending or running job.
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
- Deletes all files from the cache directory.
- **Returns**: A JSON object confirming the cache has been cleared.
