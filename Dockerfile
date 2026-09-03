FROM python:3.11-slim

WORKDIR /app

# gcc/g++ needed for LightGBM and chromadb native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Install deps in batches so each layer is cached independently.
# Heavy packages (tensorflow, sentence-transformers) get their own layer
# so a single timeout doesn't blow the whole build.

# Lightweight data + ingestion deps
RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    "duckdb>=0.10" "pandas>=2.2" "numpy>=1.26" \
    "requests>=2.31" "python-dotenv>=1.0" \
    "holidays>=0.46" "gridstatus>=0.24"

# ML — LightGBM + sklearn (moderate size)
RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    "lightgbm>=4.3" "scikit-learn>=1.4"

# TensorFlow (large — isolated so failure here doesn't re-run above layers)
RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    "tensorflow>=2.16"

# Serving + agent deps
RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    "langgraph>=0.1" "fastapi>=0.111" "uvicorn>=0.29" \
    "streamlit>=1.35" "plotly>=5.0" "anthropic>=0.40" \
    "pypdf>=4.0"

# sentence-transformers + chromadb (large models downloaded at runtime)
RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    "chromadb>=0.5" "sentence-transformers>=3.0"

# Install the local package itself without re-resolving deps
COPY ingest/ ingest/
COPY models/ models/
COPY agents/ agents/
COPY api/ api/
COPY dashboard/ dashboard/

RUN pip install --no-cache-dir --no-deps -e .

ENV GRIDPULSE_DB=/app/gridpulse.duckdb

EXPOSE 8000 8501
