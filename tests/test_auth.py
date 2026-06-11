"""Optional bearer auth: enforced when MEMORY_AUTH_TOKEN is set, off otherwise."""

import pytest
from fastapi.testclient import TestClient

from conftest import make_turn
from memory_service import config, db
from memory_service.app import app


@pytest.fixture()
def auth_client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(config, "AUTH_TOKEN", "sekrit")
    db.close()
    db.init(str(tmp_path / "auth.db"))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    db.close()


def test_health_open_without_token(auth_client):
    assert auth_client.get("/health").status_code == 200


def test_endpoints_reject_missing_or_wrong_token(auth_client):
    assert auth_client.post("/turns", json=make_turn()).status_code == 401
    r = auth_client.post("/turns", json=make_turn(),
                         headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_endpoints_accept_correct_token(auth_client):
    headers = {"Authorization": "Bearer sekrit"}
    assert auth_client.post("/turns", json=make_turn(), headers=headers).status_code == 201
    assert auth_client.post("/recall", json={"query": "x", "session_id": "s", "user_id": "u1",
                                             "max_tokens": 128}, headers=headers).status_code == 200
