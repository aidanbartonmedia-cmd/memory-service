"""Retrieval: keyword (FTS5/BM25) search. v0.1 — keyword-only baseline.

Dense + fusion lands in a later iteration (see CHANGELOG).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from . import db

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


def retrieve(owner: str, query: str, *, limit: int = 12) -> dict[str, list[tuple[str, float]]]:
    """v0.1 baseline: BM25 only. Returns {"memories": [...], "turns": [...]}."""
    return {
        "memories": bm25_memories(owner, query, limit),
        "turns": bm25_turns(owner, query, limit),
    }
