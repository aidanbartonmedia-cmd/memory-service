"""Restart persistence, concurrent-session isolation, supersession chains."""

import threading

from conftest import make_turn

from memory_service import db


def test_restart_persistence(client, db_path):
    client.post("/turns", json=make_turn(text="I work at Stripe in Seattle."))
    # Simulate a process restart: tear down the connection, reopen the file.
    db.close()
    db.init(db_path)
    r = client.post("/recall", json={"query": "Where does the user work?",
                                     "session_id": "s2", "user_id": "u1", "max_tokens": 512})
    assert "Stripe" in r.json()["context"]
    mems = client.get("/users/u1/memories").json()["memories"]
    assert any("Stripe" in m["value"] for m in mems)


def test_concurrent_sessions_do_not_bleed(client):
    """Two users hammering /turns in parallel; neither sees the other's facts."""
    errors: list[Exception] = []

    def ingest(user: str, employer: str, city: str):
        try:
            for i in range(5):
                r = client.post("/turns", json=make_turn(
                    session_id=f"{user}-s{i}", user_id=user,
                    text=f"Reminder {i}: I work at {employer} and I live in {city}.",
                ))
                assert r.status_code == 201
        except Exception as e:  # surface thread failures in the main thread
            errors.append(e)

    t1 = threading.Thread(target=ingest, args=("alice", "Acme", "Oslo"))
    t2 = threading.Thread(target=ingest, args=("bob", "Globex", "Lima"))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors

    r_alice = client.post("/recall", json={"query": "Where does the user work and live?",
                                           "session_id": "probe-a", "user_id": "alice",
                                           "max_tokens": 512}).json()
    r_bob = client.post("/recall", json={"query": "Where does the user work and live?",
                                         "session_id": "probe-b", "user_id": "bob",
                                         "max_tokens": 512}).json()
    assert "Acme" in r_alice["context"] and "Globex" not in r_alice["context"]
    assert "Globex" in r_bob["context"] and "Acme" not in r_bob["context"]


def test_anonymous_sessions_isolated_from_each_other(client):
    client.post("/turns", json=make_turn(session_id="anon-1", user_id=None,
                                         text="I work at SecretCorp."))
    r = client.post("/recall", json={"query": "Where does the user work?",
                                     "session_id": "anon-2", "user_id": None,
                                     "max_tokens": 256})
    assert "SecretCorp" not in r.json()["context"]


def test_same_user_shares_across_sessions(client):
    client.post("/turns", json=make_turn(session_id="sess-a", text="I work at Stripe."))
    r = client.post("/recall", json={"query": "Where does the user work?",
                                     "session_id": "sess-b", "user_id": "u1",
                                     "max_tokens": 256})
    assert "Stripe" in r.json()["context"]


def test_recall_with_null_user_resolves_session_owner(client):
    """Turns written with a user_id must be recallable when the caller only
    knows the session_id (user_id: null in the recall body)."""
    client.post("/turns", json=make_turn(session_id="sess-x", user_id="carol",
                                         text="I work at Initech."))
    r = client.post("/recall", json={"query": "Where does the user work?",
                                     "session_id": "sess-x", "user_id": None,
                                     "max_tokens": 256})
    assert "Initech" in r.json()["context"]


def test_supersession_chain(client):
    client.post("/turns", json=make_turn(session_id="s1", timestamp="2025-01-01T00:00:00Z",
                                         text="I work at Stripe."))
    client.post("/turns", json=make_turn(session_id="s3", timestamp="2025-04-01T00:00:00Z",
                                         text="I just started at Notion."))
    mems = client.get("/users/u1/memories").json()["memories"]
    actives = [m for m in mems if m["active"] and "employ" in m["key"]]
    inactive = [m for m in mems if not m["active"] and "employ" in m["key"]]
    assert len(actives) == 1 and "Notion" in actives[0]["value"]
    assert len(inactive) == 1 and "Stripe" in inactive[0]["value"]
    assert actives[0]["supersedes"] == inactive[0]["id"]
    assert inactive[0]["superseded_by"] == actives[0]["id"]

    r = client.post("/recall", json={"query": "Where does the user work?",
                                     "session_id": "probe", "user_id": "u1",
                                     "max_tokens": 512}).json()
    first_line = next(l for l in r["context"].splitlines() if "Notion" in l or "Stripe" in l)
    assert "Notion" in first_line, "recall must lead with the current fact"


def test_delete_session_repairs_chain(client):
    client.post("/turns", json=make_turn(session_id="s1", text="I work at Stripe."))
    client.post("/turns", json=make_turn(session_id="s2", text="I just started at Notion."))
    client.delete("/sessions/s2")
    mems = client.get("/users/u1/memories").json()["memories"]
    actives = [m for m in mems if m["active"] and "employ" in m["key"]]
    assert len(actives) == 1 and "Stripe" in actives[0]["value"], \
        "deleting the superseding session must reactivate the prior fact"


def test_budget_respected(client):
    for i in range(8):
        client.post("/turns", json=make_turn(
            session_id=f"s{i}",
            text=f"Fact number {i}: I really enjoy hobby number {i}, it is great.",
        ))
    for budget in (32, 64, 128):
        r = client.post("/recall", json={"query": "what are the user's hobbies?",
                                         "session_id": "probe", "user_id": "u1",
                                         "max_tokens": budget})
        ctx = r.json()["context"]
        assert len(ctx) // 4 <= budget * 2, f"blew 2x budget at max_tokens={budget}"
