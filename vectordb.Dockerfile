# Start from a Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    gcc \
    g++ \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
RUN pip install uv

RUN uv pip install --system \
    "git+https://github.com/lamalab-org/MoleculeBind.git" \
    "fastapi" \
    "uvicorn[standard]" \
    "python-multipart"

# Copy only your application code
COPY ./api_vectordb.py ./api_vectordb.py
COPY ./retrieval.py ./retrieval.py
COPY ./prune.py ./prune.py
COPY ./filtered_pubchem.parquet ./filtered_pubchem.parquet

# Expose the port the API will run on
EXPOSE 7999

# Command to run the API server
CMD ["uvicorn", "api_vectordb:app", "--host", "0.0.0.0", "--port", "7999"]
