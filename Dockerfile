# ---- Build stage: install deps into a venv ----
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage: copy venv + app, run as non-root ----
FROM python:3.11-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*    # LightGBM needs OpenMP at runtime
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ ./src/
COPY conf/ ./conf/
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e . --no-deps   # install package, deps already in venv

# Model + feature-store source must be present at runtime.
COPY data/04_models/ ./data/04_models/
COPY data/02_intermediate/ ./data/02_intermediate/

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "fraud_detection.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]