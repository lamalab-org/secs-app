import requests
import json
import time
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def interpolate_spectrum(x, y, x_min=-2, x_max=10, num_points=10000, y_normalize=True):
    """
    Interpolate and normalize NMR spectrum data.

    Parameters:
    -----------
    x : array-like
        Chemical shift values (ppm)
    y : array-like
        Intensity values
    x_min : float, default=-2
        Minimum x value for interpolation range (ppm)
    x_max : float, default=10
        Maximum x value for interpolation range (ppm)
    num_points : int, default=1000
        Number of points in the interpolated spectrum
    y_normalize : bool, default=True
        Whether to normalize y values to [0, 1] range

    Returns:
    --------
    dict: Dictionary containing 'x' and 'y' arrays for interpolated spectrum
    """

    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Sort by x values (important for interpolation)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Remove any duplicate x values (keep the mean y value)
    unique_x, unique_idx = np.unique(x_sorted, return_inverse=True)
    if len(unique_x) < len(x_sorted):
        unique_y = np.array(
            [np.mean(y_sorted[unique_idx == i]) for i in range(len(unique_x))]
        )
    else:
        unique_y = y_sorted
        unique_x = x_sorted

    # Create interpolation function
    # Use 'bounds_error=False' and 'fill_value=0' to handle extrapolation
    f = interp1d(unique_x, unique_y, kind="linear", bounds_error=False, fill_value=0)

    # Create new x range
    x_new = np.linspace(x_min, x_max, num_points)

    # Interpolate y values
    y_new = f(x_new)

    # Normalize y values to [0, 1] if requested
    if y_normalize:
        y_min, y_max = np.min(y_new), np.max(y_new)
        if y_max > y_min:  # Avoid division by zero
            y_new = (y_new - y_min) / (y_max - y_min)
        else:
            y_new = np.zeros_like(y_new)

    return {"x": x_new.tolist(), "y": y_new.tolist()}


# Example usage with your existing code:
def process_spectrum_for_payload(spectrum_df):
    """
    Process spectrum dataframe for API payload.
    """
    x = spectrum_df.x.tolist()
    y = spectrum_df.y.tolist()

    # Interpolate and normalize
    interpolated = interpolate_spectrum(x, y)

    return interpolated


spectrum = pd.read_csv("test.tsv", sep="\t")


payload = {
    "mf": "C4H8O",
    "spectrum": process_spectrum_for_payload(spectrum),
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
