import requests
import json
import time
import pandas as pd
# Define the request payload with smaller GA parameters for a quicker test

spectrum = pd.read_csv("test.tsv", sep="\t")
x = spectrum.x.tolist()
y = spectrum.y.tolist()

payload = {
    "mf": "C4H8O",
    "spectrum": {"x": x, "y": y},
    "pop_ga": 50,
    "offspring_ga": 256,
    "gens_ga": 5,
    # "initial_environment": {"reactants": "c1ccccc1", "reagents": ""},
}

API_BASE_URL = "http://localhost:8000"

try:
    # 1. Submit the job
    print("Submitting job to the API...")
    response = requests.post(f"{API_BASE_URL}/submit", json=payload)
    response.raise_for_status()

    response_json = response.json()
    job_id = response_json.get("job_id")

    if not job_id:
        print("Error: 'job_id' not found in the submission response.")
        print("Response content:", response_json)
        exit()

    print(f"Job submitted successfully. Job ID: {job_id}")
    print("-" * 40)

    # 2. Poll for job status
    while True:
        print(f"Checking status for job {job_id}...")
        status_response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/status")
        status_response.raise_for_status()
        status_data = status_response.json()

        status = status_data.get("status", "unknown").lower()
        print(f"Current job status: {status}")

        if status == "success":
            print("Job finished successfully. Fetching results...")
            break
        elif status in ["failure", "failed"]:
            print("Job failed.")
            print("Error details:", json.dumps(status_data, indent=2))
            exit()

        # Wait before polling again
        time.sleep(5)

    print("-" * 40)

    # 3. Fetch the final result
    print("Fetching final result...")
    result_response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/result")
    result_response.raise_for_status()

    print("Final Result:")
    print(json.dumps(result_response.json(), indent=2))

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the request: {e}")
