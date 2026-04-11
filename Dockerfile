# ── Stage 1: base image ───────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps (needed for some numpy builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: install Python dependencies ──────────────────────────────────────
COPY pyproject.toml ./

# Install the project dependencies from pyproject.toml
RUN pip install --upgrade pip && \
    pip install "numpy>=1.26.0" \
                "fastapi>=0.110.0" \
                "uvicorn[standard]>=0.29.0" \
                "pydantic>=2.6.0" \
                "python-dotenv>=1.0.0" \
                "requests>=1.0.0" \
                "openai>=1.0.0"

# ── Stage 3: copy source code ─────────────────────────────────────────────────
COPY app/    ./app/
COPY server/ ./server/
COPY run.sh  ./run.sh
COPY inference.py ./*.txt ./

# Make run script executable
RUN chmod +x ./run.sh

# ── Stage 4: expose & run ─────────────────────────────────────────────────────
EXPOSE 7860

CMD ["/bin/bash", "./run.sh"]
