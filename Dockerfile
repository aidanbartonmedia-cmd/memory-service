# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed-cache

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image: runtime needs no network access
# and no API key for dense retrieval, and cold start is fast.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY src/ src/

# /data is the named-volume mount point for the SQLite store.
RUN useradd --create-home appuser && mkdir -p /data && chown -R appuser:appuser /data /app
USER appuser

ENV PYTHONPATH=/app/src
EXPOSE 8080

HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=10 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).status == 200 else 1)"]

CMD ["uvicorn", "memory_service.app:app", "--host", "0.0.0.0", "--port", "8080"]
