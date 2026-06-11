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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


# ---------- turns ----------

def add_turn(
    *,
    session_id: str,
    user_id: str | None,
    ts: str,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> str:
    turn_id = new_id("turn")
    owner = owner_key(user_id, session_id)
    fts_text = " \n".join(str(m.get("content", "")) for m in messages)
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO turns (id, session_id, user_id, owner, ts, messages_json,"
            " metadata_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                turn_id,
                session_id,
                user_id,
                owner,
                ts,
                json.dumps(messages, ensure_ascii=False),
                json.dumps(metadata or {}, ensure_ascii=False),
                _now(),
            ),
        )
        conn.execute(
            "INSERT INTO turns_fts (doc_id, text) VALUES (?,?)", (turn_id, fts_text)
        )
    return turn_id


def enrich_turn(turn_id: str, *, summary: str, embedding: bytes | None) -> None:
    with db.tx() as conn:
        conn.execute(
            "UPDATE turns SET summary=?, embedding=? WHERE id=?",
            (summary, embedding, turn_id),
        )
        if summary:
            # The summary is a much better retrieval target than raw chat text;
            # index both (raw text already inserted at add_turn).
            conn.execute(
                "INSERT INTO turns_fts (doc_id, text) VALUES (?,?)", (turn_id, summary)
            )


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


# ---------- memories ----------

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
    mem_id = new_id("mem")
    now = _now()
    with db.tx() as conn:
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


def touch_memory(mem_id: str, *, confidence: float | None = None) -> None:
    """Re-affirmed memory: bump updated_at (and optionally confidence)."""
    with db.tx() as conn:
        if confidence is not None:
            conn.execute(
                "UPDATE memories SET updated_at=?, confidence=? WHERE id=?",
                (_now(), confidence, mem_id),
            )
        else:
            conn.execute("UPDATE memories SET updated_at=? WHERE id=?", (_now(), mem_id))


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
    """If a deleted memory superseded an older one, reactivate the older one
    (unless it was itself deleted). Keeps chains consistent after cleanup."""
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
