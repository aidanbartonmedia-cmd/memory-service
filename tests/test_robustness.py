"""Malformed input, unicode oddities, oversized payloads: 4xx, never a crash."""

from conftest import make_turn


def test_bad_json_4xx(client):
    r = client.post("/turns", content=b"{not json", headers={"Content-Type": "application/json"})
    assert 400 <= r.status_code < 500
    assert client.get("/health").status_code == 200


def test_missing_fields_4xx(client):
    assert 400 <= client.post("/turns", json={}).status_code < 500
    assert 400 <= client.post("/turns", json={"session_id": "s"}).status_code < 500
    assert 400 <= client.post("/turns", json={"session_id": "s", "messages": []}).status_code < 500


def test_wrong_types_4xx(client):
    r = client.post("/turns", json={"session_id": 12.5, "messages": "nope"})
    assert 400 <= r.status_code < 500
    r = client.post("/recall", json={"query": "x", "session_id": "s", "max_tokens": "many"})
    assert 400 <= r.status_code < 500
    r = client.post("/recall", json={"query": "x", "session_id": "s", "max_tokens": -5})
    assert 400 <= r.status_code < 500


def test_unicode_oddities_survive(client):
    weird = "👩‍👩‍👧‍👦 ر سالة नमस्ते ‮ REVERSED \x00� ZALGO H̸̡̪̯ͨ͊̽̅̾̎ẽ̴̢̢̟͈͖̈ l̶lo"
    r = client.post("/turns", json=make_turn(text=weird))
    assert r.status_code == 201
    r = client.post("/recall", json={"query": "नमस्ते 👩‍👩‍👧‍👦", "session_id": "s1",
                                     "user_id": "u1", "max_tokens": 256})
    assert r.status_code == 200
    r = client.post("/search", json={"query": weird, "user_id": "u1", "limit": 3})
    assert r.status_code == 200


def test_fts_metacharacters_safe(client):
    client.post("/turns", json=make_turn(text="I like SQL injection jokes"))
    for q in ['"unbalanced', "a AND OR NOT (", "col:val*", "'; DROP TABLE--", "(((((", "*"]:
        r = client.post("/recall", json={"query": q, "session_id": "s1",
                                         "user_id": "u1", "max_tokens": 128})
        assert r.status_code == 200, q


def test_oversized_payload_413(client):
    big = "x" * (9 * 1024 * 1024)
    r = client.post("/turns", json=make_turn(text=big))
    assert r.status_code == 413


def test_giant_but_legal_message_ok(client):
    big = "I love hiking. " * 20_000  # ~300KB, inside the body cap
    r = client.post("/turns", json=make_turn(text=big))
    assert r.status_code == 201


def test_extra_fields_tolerated(client):
    turn = make_turn()
    turn["future_field"] = {"nested": True}
    turn["messages"][0]["tool_call_id"] = "tc_1"
    assert client.post("/turns", json=turn).status_code == 201


def test_tool_role_messages(client):
    turn = {
        "session_id": "s-tool",
        "user_id": "u-tool",
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "tool", "name": "get_weather", "content": '{"temp": 38}'},
            {"role": "assistant", "content": "It's 38F."},
        ],
        "timestamp": "2025-03-15T10:30:00Z",
        "metadata": {},
    }
    assert client.post("/turns", json=turn).status_code == 201


def test_non_string_content_coerced(client):
    turn = make_turn()
    turn["messages"][0]["content"] = {"blocks": [1, 2, 3]}
    assert client.post("/turns", json=turn).status_code == 201


def test_unknown_routes_404_not_crash(client):
    assert client.get("/nope").status_code == 404
    assert client.post("/turns/extra", json={}).status_code in (404, 405)
