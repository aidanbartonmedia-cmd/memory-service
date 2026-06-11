"""Environment-driven configuration. Every knob has a working default."""

from __future__ import annotations

import os
from pathlib import Path


def _default_db_path() -> str:
    # /data is the Docker volume mount; fall back to a local dir for dev.
    if Path("/data").is_dir() and os.access("/data", os.W_OK):
        return "/data/memory.db"
    local = Path(__file__).resolve().parents[2] / "data"
    local.mkdir(exist_ok=True)
    return str(local / "memory.db")


DB_PATH: str = os.environ.get("MEMORY_DB_PATH", _default_db_path())

# Optional bearer auth. If unset, all requests are accepted.
AUTH_TOKEN: str | None = os.environ.get("MEMORY_AUTH_TOKEN") or None

# LLM extraction. If no key is present the service falls back to a
# rule-based extractor (documented in README: Failure modes).
ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY") or None
LLM_MODEL: str = os.environ.get("MEMORY_LLM_MODEL", "claude-opus-4-8")
# The eval gives /turns 60 seconds TOTAL. SDK retries compose multiplicatively
# (and 429 retry-after sleeps can exceed the per-attempt timeout), so the
# per-attempt knobs are kept tight AND llm_extract enforces a hard request-
# level wall-clock deadline, after which extraction falls back to heuristics.
LLM_MAX_RETRIES: int = int(os.environ.get("MEMORY_LLM_MAX_RETRIES", "1"))
LLM_TIMEOUT_S: float = float(os.environ.get("MEMORY_LLM_TIMEOUT_S", "20"))
EXTRACTION_DEADLINE_S: float = float(os.environ.get("MEMORY_EXTRACTION_DEADLINE_S", "40"))

# Local embedding model (ONNX via fastembed; baked into the Docker image).
EMBEDDING_MODEL: str = os.environ.get("MEMORY_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Retrieval tuning (calibrated against fixtures/ -- see CHANGELOG v0.4 for
# the calibration data behind the two floors).
RRF_K: int = int(os.environ.get("MEMORY_RRF_K", "60"))
DENSE_FLOOR: float = float(os.environ.get("MEMORY_DENSE_FLOOR", "0.62"))
DENSE_FLOOR_LOW: float = float(os.environ.get("MEMORY_DENSE_FLOOR_LOW", "0.50"))
TERM_FLOOR: float = float(os.environ.get("MEMORY_TERM_FLOOR", "0.52"))
HOP_DAMPING: float = float(os.environ.get("MEMORY_HOP_DAMPING", "0.5"))

# Resilience
MAX_BODY_BYTES: int = int(os.environ.get("MEMORY_MAX_BODY_BYTES", str(8 * 1024 * 1024)))
