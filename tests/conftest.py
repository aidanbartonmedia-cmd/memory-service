"""Test fixtures.

Tests run with LLM extraction DISABLED (heuristic fallback mode) so the suite
is fast, free, deterministic, and runs without network or API keys. The
LLM-quality path is exercised separately by scripts/selfeval.py against a
running service (see README: How to run the tests).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory_service import config, db  # noqa: E402
from memory_service.app import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(config, "AUTH_TOKEN", None)
    db.close()
    db.init(str(tmp_path / "test.db"))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    db.close()


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "test.db")


def make_turn(session_id="s1", user_id="u1", text="Hello there", assistant="Hi!", **kw):
    return {
        "session_id": session_id,
        "user_id": user_id,
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": assistant},
        ],
        "timestamp": kw.get("timestamp", "2025-03-15T10:30:00Z"),
        "metadata": kw.get("metadata", {}),
    }
