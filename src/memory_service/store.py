"""Persistence operations for turns and memories.

Owner scoping: every row carries an `owner` key — the user_id when present,
otherwise "session:<session_id>". Memories are deliberately shared across
sessions for the same user_id (documented in README); anonymous turns are
scoped to their session so concurrent anonymous sessions can never bleed.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from . import db

log = logging.getLogger("memory.store")


def owner_key(user_id: str | None, session_id: str | None) -> str:
    if user_id:
        return user_id
    return f"session:{session_id or 'unknown'}"


def resolve_owner(user_id: str | None, session_id: str | None) -> str:
    """Owner for READ paths. A recall with user_id null but a session_id whose
    turns were written under a user must reach that user's memories — the
    caller knows the session, not necessarily the user. Falls back to the
    anonymous session scope only when the session has no known user."""
    if user_id:
        return user_id
    if session_id:
        with db.tx() as conn:
            row = conn.execute(
                "SELECT user_id FROM turns WHERE session_id=? AND user_id IS NOT NULL"
                " ORDER BY ts DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row and row["user_id"]:
            return row["user_id"]
    return owner_key(None, session_id)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


# ---------- turns ----------

def get_turns(owner: str, limit: int = 50) -> list[dict[str, Any]]:
    with db.tx() as conn:
        rows = conn.execute(
            "SELECT * FROM turns WHERE owner=? ORDER BY ts DESC LIMIT ?", (owner, limit)
        ).fetchall()
    return [dict(r) for r in rows]


def get_turn(turn_id: str) -> dict[str, Any] | None:
    with db.tx() as conn:
        row = conn.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
    return dict(row) if row else None


# ---------- atomic turn write (turn + memories + enrichment in one tx) ----------

def write_turn(
    *,
    session_id: str,
    user_id: str | None,
    ts: str,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    summary: str,
    turn_embedding: bytes | None,
    memory_ops: list[dict[str, Any]],
) -> tuple[str, int]:
    """Apply a fully-extracted turn in a single transaction.

    The LLM extraction happens *before* this call (lock-free); here we only
    do fast SQL. Atomicity is the point: a kill mid-write rolls everything
    back — the eval's restart-mid-write probe can never observe a turn
    without its memories or vice versa. memory_ops entries carry the
    ExtractedMemory fields plus a precomputed "embedding" blob.
    """
    turn_id = new_id("turn")
    owner = owner_key(user_id, session_id)
    fts_text = " \n".join(str(m.get("content", "")) for m in messages)
    applied = 0
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO turns (id, session_id, user_id, owner, ts, messages_json,"
            " metadata_json, summary, embedding, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                turn_id, session_id, user_id, owner, ts,
                json.dumps(messages, ensure_ascii=False),
                json.dumps(metadata or {}, ensure_ascii=False),
                summary, turn_embedding, _now(),
            ),
        )
        conn.execute("INSERT INTO turns_fts (doc_id, text) VALUES (?,?)", (turn_id, fts_text))
        if summary:
            conn.execute("INSERT INTO turns_fts (doc_id, text) VALUES (?,?)", (turn_id, summary))
        for op in memory_ops:
            try:
                if op["action"] == "reinforce" and op.get("supersedes_id"):
                    # Guarded: target must exist, belong to this owner, and be
                    # active (reinforcing a superseded row would corrupt the
                    # chain the eval inspects); confidence only ratchets up —
                    # a low-confidence heuristic restatement must not erode a
                    # high-confidence LLM fact.
                    conn.execute(
                        "UPDATE memories SET updated_at=?,"
                        " confidence=MAX(confidence, ?)"
                        " WHERE id=? AND owner=? AND active=1",
                        (_now(), op["confidence"], op["supersedes_id"], owner),
                    )
                    continue
                _insert_memory_conn(
                    conn,
                    owner=owner,
                    user_id=user_id,
                    type_=op["type"],
                    key=op["key"],
                    value=op["value"],
                    confidence=op["confidence"],
                    entities=op.get("entities", []),
                    source_session=session_id,
                    source_turn=turn_id,
                    supersedes_id=op.get("supersedes_id"),
                    embedding=op.get("embedding"),
                )
                applied += 1
            except Exception:
                log.exception("failed to apply extracted memory %s", op.get("key"))
    return turn_id, applied


# ---------- memories ----------

def _insert_memory_conn(
    conn: Any,
    *,
    owner: str,
    user_id: str | None,
    type_: str,
    key: str,
    value: str,
    confidence: float,
    entities: list[str],
    source_session: str | None,
    source_turn: str | None,
    supersedes_id: str | None,
    embedding: bytes | None,
) -> str:
    mem_id = new_id("mem")
    now = _now()
    if supersedes_id:
        row = conn.execute(
            "SELECT id FROM memories WHERE id=? AND owner=?", (supersedes_id, owner)
        ).fetchone()
        if row is None:
            log.warning("supersedes target %s not found for owner %s", supersedes_id, owner)
            supersedes_id = None
        else:
            conn.execute(
                "UPDATE memories SET active=0, superseded_by=?, updated_at=? WHERE id=?",
                (mem_id, now, supersedes_id),
            )
    conn.execute(
        "INSERT INTO memories (id, owner, user_id, type, key, value, confidence,"
        " entities_json, source_session, source_turn, created_at, updated_at,"
        " supersedes, active, embedding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)",
        (
            mem_id, owner, user_id, type_, key, value, confidence,
            json.dumps(sorted(set(e.lower() for e in entities)), ensure_ascii=False),
            source_session, source_turn, now, now, supersedes_id, embedding,
        ),
    )
    conn.execute(
        "INSERT INTO memories_fts (doc_id, text) VALUES (?,?)",
        (mem_id, f"{key} {value} {' '.join(entities)}"),
    )
    return mem_id


def insert_memory(
    *,
    owner: str,
    user_id: str | None,
    type_: str,
    key: str,
    value: str,
    confidence: float,
    entities: list[str],
    source_session: str | None,
    source_turn: str | None,
    supersedes_id: str | None = None,
    embedding: bytes | None = None,
) -> str:
    with db.tx() as conn:
        return _insert_memory_conn(
            conn, owner=owner, user_id=user_id, type_=type_, key=key, value=value,
            confidence=confidence, entities=entities, source_session=source_session,
            source_turn=source_turn, supersedes_id=supersedes_id, embedding=embedding,
        )


def get_memories(owner: str, *, active_only: bool = False) -> list[dict[str, Any]]:
    q = "SELECT * FROM memories WHERE owner=?"
    if active_only:
        q += " AND active=1"
    q += " ORDER BY created_at ASC"
    with db.tx() as conn:
        rows = conn.execute(q, (owner,)).fetchall()
    return [dict(r) for r in rows]


def get_memory(mem_id: str) -> dict[str, Any] | None:
    with db.tx() as conn:
        row = conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
    return dict(row) if row else None


# ---------- deletes (eval cleanup) ----------

def _repair_supersession(conn: Any, deleted_ids: set[str]) -> None:
    """Keep chains consistent after cleanup deletes, both directions:
    - a survivor that was superseded BY a deleted memory is reactivated
      (the contradiction that retired it no longer exists);
    - a survivor whose `supersedes` points AT a deleted memory gets the
      pointer nulled (no dangling ids in the inspectable chain)."""
    if not deleted_ids:
        return
    marks = ",".join("?" for _ in deleted_ids)
    rows = conn.execute(
        f"SELECT id FROM memories WHERE superseded_by IN ({marks})", tuple(deleted_ids)
    ).fetchall()
    for r in rows:
        conn.execute(
            "UPDATE memories SET active=1, superseded_by=NULL, updated_at=? WHERE id=?",
            (_now(), r["id"]),
        )
    conn.execute(
        f"UPDATE memories SET supersedes=NULL WHERE supersedes IN ({marks})",
        tuple(deleted_ids),
    )


def delete_session(session_id: str) -> None:
    with db.tx() as conn:
        mem_ids = {
            r["id"]
            for r in conn.execute(
                "SELECT id FROM memories WHERE source_session=? OR owner=?",
                (session_id, f"session:{session_id}"),
            ).fetchall()
        }
        turn_ids = {
            r["id"]
            for r in conn.execute(
                "SELECT id FROM turns WHERE session_id=?", (session_id,)
            ).fetchall()
        }
        for mid in mem_ids:
            conn.execute("DELETE FROM memories WHERE id=?", (mid,))
            conn.execute("DELETE FROM memories_fts WHERE doc_id=?", (mid,))
        for tid in turn_ids:
            conn.execute("DELETE FROM turns WHERE id=?", (tid,))
            conn.execute("DELETE FROM turns_fts WHERE doc_id=?", (tid,))
        _repair_supersession(conn, mem_ids)


def delete_user(user_id: str) -> None:
    with db.tx() as conn:
        mem_ids = {
            r["id"]
            for r in conn.execute(
                "SELECT id FROM memories WHERE owner=? OR user_id=?", (user_id, user_id)
            ).fetchall()
        }
        turn_ids = {
            r["id"]
            for r in conn.execute(
                "SELECT id FROM turns WHERE owner=? OR user_id=?", (user_id, user_id)
            ).fetchall()
        }
        for mid in mem_ids:
            conn.execute("DELETE FROM memories WHERE id=?", (mid,))
            conn.execute("DELETE FROM memories_fts WHERE doc_id=?", (mid,))
        for tid in turn_ids:
            conn.execute("DELETE FROM turns WHERE id=?", (tid,))
            conn.execute("DELETE FROM turns_fts WHERE doc_id=?", (tid,))
