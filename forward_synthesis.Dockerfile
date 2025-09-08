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
    "transformers" \
    "rdkit" \
    "fastapi" \
    "uvicorn[standard]" \
    "python-multipart"

RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

# Copy only your application code
COPY ./forward_synthesis.py ./forward_synthesis.py
COPY ./retrieval.py ./retrieval.py
COPY ./prune.py ./prune.py

# Expose the port the API will run on
EXPOSE 7999

# Command to run the API server
CMD ["uvicorn", "forward_synthesis:app", "--host", "0.0.0.0", "--port", "7998"]
