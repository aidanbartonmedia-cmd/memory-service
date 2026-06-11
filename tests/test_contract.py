"""Contract roundtrip: endpoint existence, shapes, status codes."""

from conftest import make_turn


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_turn_roundtrip_shapes(client):
    r = client.post("/turns", json=make_turn(text="I just moved to Berlin from NYC last month."))
    assert r.status_code == 201
    body = r.json()
    assert isinstance(body["id"], str) and body["id"]

    r = client.post("/recall", json={
        "query": "Where does this user live?",
        "session_id": "s2",
        "user_id": "u1",
        "max_tokens": 512,
    })
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"context", "citations"}
    assert isinstance(body["context"], str)
    assert "Berlin" in body["context"]
    for c in body["citations"]:
        assert set(c.keys()) == {"turn_id", "score", "snippet"}
        assert isinstance(c["score"], (int, float))


def test_search_shape(client):
    client.post("/turns", json=make_turn(text="My dog Rex loves the park."))
    r = client.post("/search", json={"query": "dog", "session_id": None, "user_id": "u1", "limit": 5})
    assert r.status_code == 200
    results = r.json()["results"]
    assert isinstance(results, list) and results
    first = results[0]
    assert set(first.keys()) == {"content", "score", "session_id", "timestamp", "metadata"}
    assert len(results) <= 5


def test_memories_endpoint_structured(client):
    client.post("/turns", json=make_turn(text="I work at Stripe and I live in Austin."))
    r = client.get("/users/u1/memories")
    assert r.status_code == 200
    memories = r.json()["memories"]
    assert memories, "extraction produced no memories"
    for m in memories:
        for field in ("id", "type", "key", "value", "confidence", "created_at",
                      "updated_at", "supersedes", "active"):
            assert field in m
        assert m["type"] in ("fact", "preference", "opinion", "event")
        # structured memories, not raw message chunks
        assert "I work at Stripe and I live in Austin." != m["value"]


def test_deletes_return_204(client):
    client.post("/turns", json=make_turn())
    assert client.delete("/sessions/s1").status_code == 204
    assert client.delete("/users/u1").status_code == 204
    # idempotent on unknown ids
    assert client.delete("/sessions/nope").status_code == 204
    assert client.delete("/users/nope").status_code == 204


def test_delete_user_removes_everything(client):
    client.post("/turns", json=make_turn(text="I work at Stripe."))
    assert client.get("/users/u1/memories").json()["memories"]
    client.delete("/users/u1")
    assert client.get("/users/u1/memories").json()["memories"] == []
    r = client.post("/recall", json={"query": "Where does the user work?",
                                     "session_id": "s9", "user_id": "u1", "max_tokens": 256})
    assert r.json()["context"] == ""


def test_cold_session_empty_not_error(client):
    r = client.post("/recall", json={"query": "What do you know about this user?",
                                     "session_id": "fresh", "user_id": "ghost", "max_tokens": 512})
    assert r.status_code == 200
    assert r.json() == {"context": "", "citations": []}


def test_anonymous_turns_allowed(client):
    r = client.post("/turns", json=make_turn(user_id=None, text="I prefer dark mode."))
    assert r.status_code == 201
    r = client.post("/recall", json={"query": "any preferences?", "session_id": "s1",
                                     "user_id": None, "max_tokens": 256})
    assert r.status_code == 200
