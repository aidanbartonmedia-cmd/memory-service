"""Local ONNX embeddings (fastembed / bge-small-en-v1.5, 384-dim).

The model is baked into the Docker image at build time, so embedding never
needs network access or an API key at runtime. If the model fails to load
(corrupt cache, exotic platform), the service degrades to keyword-only
retrieval instead of crashing — see README: Failure modes.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from . import config

log = logging.getLogger("memory.embeddings")

_model = None
_model_failed = False
_lock = threading.Lock()


def _get_model():
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    with _lock:
        if _model is not None or _model_failed:
            return _model
        try:
            from fastembed import TextEmbedding

            _model = TextEmbedding(config.EMBEDDING_MODEL)
            log.info("embedding model %s loaded", config.EMBEDDING_MODEL)
        except Exception:
            log.exception("embedding model failed to load; dense retrieval disabled")
            _model_failed = True
    return _model


def available() -> bool:
    return _get_model() is not None


def embed(texts: list[str]) -> list[np.ndarray] | None:
    """Returns L2-normalized float32 vectors, or None if the model is unavailable."""
    model = _get_model()
    if model is None or not texts:
        return None
    try:
        vecs = [np.asarray(v, dtype=np.float32) for v in model.embed(texts)]
    except Exception:
        log.exception("embedding failed")
        return None
    return [v / (np.linalg.norm(v) or 1.0) for v in vecs]


def embed_one(text: str) -> np.ndarray | None:
    out = embed([text])
    return out[0] if out else None


def to_blob(vec: np.ndarray | None) -> bytes | None:
    return vec.astype(np.float32).tobytes() if vec is not None else None


def from_blob(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def cosine_rank(query_vec: np.ndarray, items: list[tuple[str, bytes | None]]) -> list[tuple[str, float]]:
    """[(id, cosine)] best-first for items that have embeddings (all normalized)."""
    scored: list[tuple[str, float]] = []
    for item_id, blob in items:
        vec = from_blob(blob)
        if vec is None or vec.shape != query_vec.shape:
            continue
        scored.append((item_id, float(np.dot(query_vec, vec))))
    scored.sort(key=lambda x: -x[1])
    return scored
