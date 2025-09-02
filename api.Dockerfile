# Step 1: Base Image
FROM python:3.11-slim-bookworm

# Step 2: Set Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="." \
    CC="/usr/bin/gcc" \
    CXX="/usr/bin/g++" \
    POLARS_MAX_THREADS=24

# Step 3: Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    gcc \
    g++ \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Step 4: Set up the application directory
WORKDIR /app

# Step 5: Install Python Dependencies using UV
RUN pip install uv
RUN uv pip install --system --torch-backend cpu -U torch \
    "mol_ga" \
    "git+https://github.com/lamalab-org/MoleculeBind.git" \
    "pydantic" \
    "fastapi" \
    "uvicorn[standard]"

# RUN uv pip install --system --torch-backend cpu -U torch
COPY ./checkpoints/ ./checkpoints/
COPY ./retrieval.py ./retrieval.py
COPY ./main.py ./main.py
COPY ./gafuncs.py ./gafuncs.py
COPY ./prune.py ./prune.py
COPY ./app.py ./app.py
COPY ./configs ./configs/

# Step 7: Expose Port and Define Entrypoint
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]