"""SQLite layer: schema, connection management, write serialization.

Design notes
------------
- WAL mode + synchronous=FULL: every commit is durable before /turns returns,
  which is what gives us the contract's synchronous read-after-write guarantee.
- One process-wide connection guarded by an RLock. The eval workload is a few
  concurrent sessions, not thousands of QPS; serializing writes through one
  connection is simpler and strictly correct (no busy-retry loops, no
  cross-connection WAL visibility questions).
- FTS5 contentless-style side tables (doc_id UNINDEXED, text) are kept in sync
  by the store layer inside the same transaction as the main-row write.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from . import config

log = logging.getLogger("memory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    user_id       TEXT,
    owner         TEXT NOT NULL,
    ts            TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    summary       TEXT NOT NULL DEFAULT '',
    embedding     BLOB,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_owner   ON turns(owner);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

CREATE TABLE IF NOT EXISTS memories (
    id             TEXT PRIMARY KEY,
    owner          TEXT NOT NULL,
    user_id        TEXT,
    type           TEXT NOT NULL CHECK (type IN ('fact','preference','opinion','event')),
    key            TEXT NOT NULL,
    value          TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 0.8,
    entities_json  TEXT NOT NULL DEFAULT '[]',
    source_session TEXT,
    source_turn    TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    supersedes     TEXT,
    superseded_by  TEXT,
    active         INTEGER NOT NULL DEFAULT 1,
    embedding      BLOB
);
CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner, active);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(source_session);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts    USING fts5(doc_id UNINDEXED, text, tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(doc_id UNINDEXED, text, tokenize='porter unicode61');
"""

_conn: sqlite3.Connection | None = None
_lock = threading.RLock()


def init(db_path: str | None = None) -> None:
    global _conn
    path = db_path or config.DB_PATH
    with _lock:
        if _conn is not None:
            return
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        conn.commit()
        _conn = conn
        log.info("sqlite ready at %s (wal, synchronous=full)", path)


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def ready() -> bool:
    try:
        with _lock:
            if _conn is None:
                return False
            _conn.execute("SELECT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    """Serialized read/write transaction. Commits on success, rolls back on error."""
    if _conn is None:
        raise RuntimeError("db not initialized")
    with _lock:
        try:
            yield _conn
            _conn.commit()
        except Exception:
            _conn.rollback()
            raise
