"""Retrieval: hybrid keyword (FTS5/BM25) + dense (local embeddings), fused
with reciprocal rank fusion.

Why hybrid: pure embedding search misses keyword-anchored queries ("what's
their dog's name?" needs the token Biscuit-adjacent memory, not a vibe), and
pure BM25 misses paraphrases ("where is the user based?" vs "lives in
Berlin"). RRF fuses the two rankings without score calibration.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from . import config, db, embeddings

log = logging.getLogger("memory.retrieval")

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def fts_query(query: str) -> str | None:
    """Sanitize arbitrary text into a safe OR-of-terms FTS5 query."""
    terms = _TOKEN_RE.findall(query)
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms[:32])


def _bm25_candidates(fts_table: str, q: str, conn, cap: int = 400) -> dict[str, float]:
    """Raw FTS matches as {doc_id: best_negated_bm25}. bm25() can only be
    evaluated when the FTS table drives the query (SQLite flattens subqueries
    and then rejects it), so we rank here and join ownership in Python."""
    rows = conn.execute(
        f"SELECT doc_id, bm25({fts_table}) AS rank FROM {fts_table}"
        f" WHERE {fts_table} MATCH ? ORDER BY rank LIMIT ?",
        (q, cap),
    ).fetchall()
    best: dict[str, float] = {}
    for r in rows:
        score = -float(r["rank"])  # bm25() is lower-is-better; negate
        if score > best.get(r["doc_id"], float("-inf")):
            best[r["doc_id"]] = score
    return best


def bm25_memories(owner: str, query: str, limit: int = 20) -> list[tuple[str, float]]:
    """Returns [(memory_id, bm25_score)] best-first. Active memories only."""
    q = fts_query(query)
    if q is None:
        return []
    with db.tx() as conn:
        best = _bm25_candidates("memories_fts", q, conn)
        if not best:
            return []
        marks = ",".join("?" for _ in best)
        owned = {
            r["id"]
            for r in conn.execute(
                f"SELECT id FROM memories WHERE id IN ({marks}) AND owner=? AND active=1",
                (*best.keys(), owner),
            ).fetchall()
        }
    ranked = sorted(((i, s) for i, s in best.items() if i in owned), key=lambda x: -x[1])
    return ranked[:limit]


def bm25_turns(owner: str, query: str, limit: int = 20) -> list[tuple[str, float]]:
    q = fts_query(query)
    if q is None:
        return []
    with db.tx() as conn:
        best = _bm25_candidates("turns_fts", q, conn)
        if not best:
            return []
        marks = ",".join("?" for _ in best)
        owned = {
            r["id"]
            for r in conn.execute(
                f"SELECT id FROM turns WHERE id IN ({marks}) AND owner=?",
                (*best.keys(), owner),
            ).fetchall()
        }
    ranked = sorted(((i, s) for i, s in best.items() if i in owned), key=lambda x: -x[1])
    return ranked[:limit]


def dense_memories(owner: str, qvec, limit: int = 20) -> list[tuple[str, float]]:
    """[(memory_id, cosine)] over the owner's active memories."""
    with db.tx() as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM memories WHERE owner=? AND active=1 AND embedding IS NOT NULL",
            (owner,),
        ).fetchall()
    return embeddings.cosine_rank(qvec, [(r["id"], r["embedding"]) for r in rows])[:limit]


def dense_turns(owner: str, qvec, limit: int = 20) -> list[tuple[str, float]]:
    with db.tx() as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM turns WHERE owner=? AND embedding IS NOT NULL",
            (owner,),
        ).fetchall()
    return embeddings.cosine_rank(qvec, [(r["id"], r["embedding"]) for r in rows])[:limit]


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int | None = None) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1/(k + rank).
    Rank positions only — no cross-system score calibration needed."""
    k = k or config.RRF_K
    fused: dict[str, float] = {}
    for ranking in rankings:
        for pos, (doc_id, _score) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


def retrieve(owner: str, query: str, *, limit: int = 12) -> dict[str, Any]:
    """Hybrid retrieval. Returns ranked memories/turns plus diagnostics used
    by the relevance gate (max dense similarity, keyword hit counts)."""
    bm_m = bm25_memories(owner, query, limit * 2)
    bm_t = bm25_turns(owner, query, limit * 2)

    qvec = embeddings.embed_one(query) if query.strip() else None
    dn_m = dense_memories(owner, qvec, limit * 2) if qvec is not None else []
    dn_t = dense_turns(owner, qvec, limit * 2) if qvec is not None else []

    mem_ranked = rrf_fuse([bm_m, dn_m])[:limit]
    turn_ranked = rrf_fuse([bm_t, dn_t])[:limit]

    return {
        "memories": mem_ranked,
        "turns": turn_ranked,
        "diagnostics": {
            "max_dense_memory": dn_m[0][1] if dn_m else 0.0,
            "max_dense_turn": dn_t[0][1] if dn_t else 0.0,
            "bm25_memory_hits": len(bm_m),
            "bm25_turn_hits": len(bm_t),
            "dense_memory": dict(dn_m),
            "dense_turn": dict(dn_t),
        },
    }
