"""Regression tests for the degraded extraction path."""

from conftest import make_turn

from memory_service.extraction import extract, heuristic_extract


def test_heuristic_survives_user_prefixed_content_without_colon(client):
    """Multi-line content whose lines start with 'user' but contain no colon
    used to IndexError the fallback extractor and 500 the request (v0.7)."""
    turn = make_turn(text="Hi there\nusername requirements are strict here")
    r = client.post("/turns", json=turn)
    assert r.status_code == 201

    turn = make_turn(session_id="s2", text="users online: 5\nuserland tools failed")
    assert client.post("/turns", json=turn).status_code == 201


def test_heuristic_extract_handles_weird_messages_directly():
    messages = [
        {"role": "user", "content": "user\nuser user user"},
        {"role": "tool", "name": "x", "content": "userland: data"},
        {"role": "assistant", "content": ""},
    ]
    result = heuristic_extract(messages, [], "2025-01-01T00:00:00Z")
    assert result.turn_summary


def test_extract_never_raises_even_if_heuristics_break(monkeypatch):
    import memory_service.extraction as ex
    from memory_service import config

    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)

    def boom(*a, **k):
        raise RuntimeError("synthetic extractor bug")

    monkeypatch.setattr(ex, "heuristic_extract", boom)
    result, mode = extract([{"role": "user", "content": "hello"}], [], "2025-01-01T00:00:00Z")
    assert mode == "minimal"
    assert result.memories == []
    assert result.turn_summary


def test_recall_budget_clamped_not_rejected(client):
    client.post("/turns", json=make_turn(text="I work at Stripe."))
    for bad_budget in (0, -5, 1, 999999):
        r = client.post("/recall", json={"query": "where does the user work?",
                                         "session_id": "s1", "user_id": "u1",
                                         "max_tokens": bad_budget})
        assert r.status_code == 200, bad_budget


def test_search_limit_clamped_not_rejected(client):
    client.post("/turns", json=make_turn(text="I work at Stripe."))
    for bad_limit in (0, -1, 5000):
        r = client.post("/search", json={"query": "stripe", "user_id": "u1",
                                         "limit": bad_limit})
        assert r.status_code == 200, bad_limit


def test_tiny_budget_returns_bare_fact_not_empty(client):
    client.post("/turns", json=make_turn(text="I work at Stripe."))
    r = client.post("/recall", json={"query": "Where does the user work?",
                                     "session_id": "probe", "user_id": "u1",
                                     "max_tokens": 16})
    body = r.json()
    assert "Stripe" in body["context"], "tiny budgets should yield the top fact, not nothing"


def test_head_health(client):
    assert client.head("/health").status_code == 200
