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

# FTS5 indexes everything — without query-side stopword removal, "tell me
# about the user's wedding plans" keyword-matches every document containing
# "the", and keyword evidence becomes meaningless (this broke the noise gate;
# see CHANGELOG v0.4). "user" is included: our own turn summaries start with
# "User ...", so it carries zero signal.
_STOPWORDS = frozenset(
    "a an the is are was were be been being am do does did doing what what's whats "
    "who whose whom where when why how which tell me my mine your yours their theirs "
    "his her hers its our ours this that these those there here of for to in on at "
    "by with about from into onto over under and or but not no nor any some have has "
    "had having can could should would will shall may might must i you he she it we "
    "they them him us s t re ve ll d m don doesn didn isn aren wasn weren user users "
    "know knows known anything something things stuff please".split()
)


def fts_query(query: str) -> str | None:
    """Sanitize arbitrary text into a safe OR-of-content-terms FTS5 query."""
    terms = [t for t in _TOKEN_RE.findall(query) if t.lower() not in _STOPWORDS]
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


def bm25_memories(owner: str | None, query: str, limit: int = 20) -> list[tuple[str, float]]:
    """Returns [(memory_id, bm25_score)] best-first. Active memories only.
    owner=None searches globally (explicit /search with no scope)."""
    q = fts_query(query)
    if q is None:
        return []
    with db.tx() as conn:
        best = _bm25_candidates("memories_fts", q, conn)
        if not best:
            return []
        marks = ",".join("?" for _ in best)
        where = "id IN (%s) AND active=1" % marks
        params: tuple = tuple(best.keys())
        if owner is not None:
            where += " AND owner=?"
            params += (owner,)
        owned = {
            r["id"]
            for r in conn.execute(f"SELECT id FROM memories WHERE {where}", params).fetchall()
        }
    ranked = sorted(((i, s) for i, s in best.items() if i in owned), key=lambda x: -x[1])
    return ranked[:limit]


def bm25_turns(owner: str | None, query: str, limit: int = 20) -> list[tuple[str, float]]:
    q = fts_query(query)
    if q is None:
        return []
    with db.tx() as conn:
        best = _bm25_candidates("turns_fts", q, conn)
        if not best:
            return []
        marks = ",".join("?" for _ in best)
        where = "id IN (%s)" % marks
        params: tuple = tuple(best.keys())
        if owner is not None:
            where += " AND owner=?"
            params += (owner,)
        owned = {
            r["id"]
            for r in conn.execute(f"SELECT id FROM turns WHERE {where}", params).fetchall()
        }
    ranked = sorted(((i, s) for i, s in best.items() if i in owned), key=lambda x: -x[1])
    return ranked[:limit]


def dense_memories(owner: str | None, qvec, limit: int = 20) -> list[tuple[str, float]]:
    """[(memory_id, cosine)] over active memories (owner=None: global)."""
    with db.tx() as conn:
        if owner is not None:
            rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE owner=? AND active=1 AND embedding IS NOT NULL",
                (owner,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE active=1 AND embedding IS NOT NULL"
            ).fetchall()
    return embeddings.cosine_rank(qvec, [(r["id"], r["embedding"]) for r in rows])[:limit]


def dense_turns(owner: str | None, qvec, limit: int = 20) -> list[tuple[str, float]]:
    with db.tx() as conn:
        if owner is not None:
            rows = conn.execute(
                "SELECT id, embedding FROM turns WHERE owner=? AND embedding IS NOT NULL",
                (owner,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, embedding FROM turns WHERE embedding IS NOT NULL"
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


import json as _json


def entity_expand(owner: str, mem_ranked: list[tuple[str, float]], *, top_n: int = 6) -> list[tuple[str, float]]:
    """One graph hop over shared entity tags.

    'What city does the user with the dog named Biscuit live in?' retrieves
    the Biscuit memory directly; the location memory shares no tokens with the
    query and may rank poorly. Memories store entity tags at extraction time
    (['biscuit','dog','pet'] / ['denver','home','location']) — any memory
    sharing an entity with a top hit is pulled in at the parent's score damped
    by HOP_DAMPING. One hop only: damping below the relevance floor squared
    keeps second-order hops from dragging in the whole store.
    """
    if not mem_ranked:
        return mem_ranked
    parents = mem_ranked[:top_n]
    parent_score = dict(parents)
    with db.tx() as conn:
        rows = conn.execute(
            "SELECT id, entities_json FROM memories WHERE owner=? AND active=1",
            (owner,),
        ).fetchall()
    entities_of: dict[str, set[str]] = {}
    for r in rows:
        try:
            entities_of[r["id"]] = set(_json.loads(r["entities_json"] or "[]"))
        except _json.JSONDecodeError:
            entities_of[r["id"]] = set()

    parent_entities: dict[str, float] = {}
    for pid, score in parents:
        for ent in entities_of.get(pid, ()):
            parent_entities[ent] = max(parent_entities.get(ent, 0.0), score)

    ranked_ids = {i for i, _ in mem_ranked}
    expanded = list(mem_ranked)
    for mem_id, ents in entities_of.items():
        if mem_id in ranked_ids:
            continue
        shared = ents & parent_entities.keys()
        if not shared:
            continue
        hop_score = max(parent_entities[e] for e in shared) * config.HOP_DAMPING
        expanded.append((mem_id, hop_score))
    expanded.sort(key=lambda x: -x[1])
    return expanded


def _relevance_gate(diag: dict[str, Any]) -> bool:
    """Decide whether anything in the store is actually about this query.

    Calibrated on the fixture probes (see CHANGELOG v0.4): the hardest noise
    probe peaks at dense 0.585 while the weakest real probe sits at 0.592 —
    too close for a single floor. Two-tier rule instead:
      - dense >= DENSE_FLOOR (0.62): semantically close, relevant on its own
      - DENSE_FLOOR_LOW (0.50) <= dense < 0.62: ambiguous zone — require a
        stemmed-keyword (BM25) hit to confirm
      - dense < 0.50: noise, regardless of keyword collisions
    If embeddings are unavailable (degraded mode), fall back to BM25-only
    evidence — weaker noise resistance, documented in README.
    """
    max_dense = max(diag["max_dense_memory"], diag["max_dense_turn"])
    keyword_hits = diag["bm25_memory_hits"] + diag["bm25_turn_hits"]
    if diag.get("dense_available", True):
        if max_dense >= config.DENSE_FLOOR:
            return True
        return max_dense >= config.DENSE_FLOOR_LOW and keyword_hits > 0
    return keyword_hits > 0


def retrieve(owner: str | None, query: str, *, limit: int = 12) -> dict[str, Any]:
    """Hybrid retrieval + one entity hop + relevance gate.

    Returns ranked memories/turns, diagnostics, and `relevant` — when False,
    the caller returns an empty context (noise resistance)."""
    bm_m = bm25_memories(owner, query, limit * 2)
    bm_t = bm25_turns(owner, query, limit * 2)

    qvec = embeddings.embed_query(query) if query.strip() else None
    dn_m = dense_memories(owner, qvec, limit * 2) if qvec is not None else []
    dn_t = dense_turns(owner, qvec, limit * 2) if qvec is not None else []

    mem_ranked = rrf_fuse([bm_m, dn_m])
    if owner is not None:
        mem_ranked = entity_expand(owner, mem_ranked)
    mem_ranked = mem_ranked[:limit]
    turn_ranked = rrf_fuse([bm_t, dn_t])[:limit]

    diagnostics = {
        "max_dense_memory": dn_m[0][1] if dn_m else 0.0,
        "max_dense_turn": dn_t[0][1] if dn_t else 0.0,
        "bm25_memory_hits": len(bm_m),
        "bm25_turn_hits": len(bm_t),
        "dense_available": qvec is not None,
        "dense_memory": dict(dn_m),
        "dense_turn": dict(dn_t),
    }
    return {
        "memories": mem_ranked,
        "turns": turn_ranked,
        "relevant": _relevance_gate(diagnostics) if query.strip() else True,
        "diagnostics": diagnostics,
    }
