# Dockerfile — reproducible environment for Render
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps needed by some ML packages (add/remove as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl libopenblas-dev libsndfile1 libjpeg-dev zlib1g-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better layer caching
COPY pyproject.toml setup.cfg ./
COPY src ./src
COPY README.md ./
COPY . .

RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU wheels from official index first (adjust versions if needed)
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install package in editable mode (this will install the rest of dependencies)
RUN pip install --no-cache-dir -e .

# Expose data dir as a volume and port
VOLUME ["/data"]
ENV PORT=8000
ENV WORKERS=1

# Use proxy-headers in case Render proxies requests
CMD ["sh", "-c", "uvicorn ss_ai_server.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1} --proxy-headers"]
