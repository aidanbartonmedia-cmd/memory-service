"""Retraction (cessation without replacement) and supersession-history rendering.

Retraction is modeled as supersession by the negated CURRENT state ("No longer
has a dog"), never bare deactivation: recall context must let a frozen LLM
distinguish "never had a pet" from "no longer has one" (README §5). These run
in heuristic mode; the LLM path is covered by fixture probes p33/p34 via
scripts/selfeval.py.
"""

from conftest import make_turn

from memory_service.extraction import heuristic_extract

TS = "2025-05-08T19:00:00Z"


def test_heuristic_pet_retraction_supersedes_to_negated_state():
    existing = [{"id": "mem_1", "type": "fact", "key": "pets.cat.name",
                 "value": "Has a cat named Mochi"}]
    result = heuristic_extract(
        [{"role": "user", "content": "We gave the cat away last week, sadly."}],
        existing, TS,
    )
    ops = {m.key: m for m in result.memories}
    assert "pets.cat.name" in ops
    assert ops["pets.cat.name"].action == "supersede"
    assert ops["pets.cat.name"].supersedes_id == "mem_1"
    assert "no longer" in ops["pets.cat.name"].value.lower()


def test_heuristic_diet_retraction():
    existing = [{"id": "mem_2", "type": "preference", "key": "diet.style",
                 "value": "Is vegetarian"}]
    result = heuristic_extract(
        [{"role": "user", "content": "I'm not vegetarian anymore, I eat fish now."}],
        existing, TS,
    )
    ops = {m.key: m for m in result.memories}
    assert ops["diet.style"].action == "supersede"
    assert ops["diet.style"].supersedes_id == "mem_2"
    assert "no longer" in ops["diet.style"].value.lower()


def test_heuristic_retraction_never_fires_on_idioms_or_adjacent_topics():
    """Review findings: each of these matched a v0.8 draft pattern and would
    have destructively superseded a TRUE fact. None may retract anything."""
    existing = [
        {"id": "m1", "type": "fact", "key": "pets.dog.name", "value": "Has a dog named Biscuit"},
        {"id": "m2", "type": "fact", "key": "pets.cat.name", "value": "Has a cat named Mochi"},
        {"id": "m3", "type": "preference", "key": "diet.style", "value": "Is vegetarian"},
    ]
    for text in (
        "I don't have a dog in this fight, just sharing my opinion.",
        "I don't have my dog with me today, he's at the groomer.",
        "We gave the dog food away to a neighbor.",
        "I gave my cat toys away to the shelter.",
        "I'm not a vegetarian-hater, I just like steak.",
        "I am not a vegan chef, just a home cook.",
    ):
        result = heuristic_extract([{"role": "user", "content": text}], existing, TS)
        retractions = [m for m in result.memories if "no longer" in m.value.lower()]
        assert not retractions, f"false retraction on: {text!r} -> {retractions}"


def test_heuristic_replacement_after_retraction_wins_by_text_position():
    """Review finding: 'I left Stripe. I joined Notion last week.' must yield
    the replacement, not the retraction — pattern-list ordering used to let
    the retraction claim the key and silently drop Notion."""
    existing = [{"id": "m1", "type": "fact", "key": "employment.employer",
                 "value": "Works at Stripe"}]
    result = heuristic_extract(
        [{"role": "user", "content": "I left Stripe. I joined Notion last week."}],
        existing, TS,
    )
    ops = {m.key: m for m in result.memories}
    assert ops["employment.employer"].value == "Works at Notion"
    assert ops["employment.employer"].action == "supersede"

    result = heuristic_extract(
        [{"role": "user", "content": "I'm not vegetarian anymore, I'm vegan now."}],
        [{"id": "m2", "type": "preference", "key": "diet.style", "value": "Is vegetarian"}],
        TS,
    )
    ops = {m.key: m for m in result.memories}
    assert ops["diet.style"].value == "Is vegan"


def test_heuristic_rule13_comma_example_does_not_retract():
    """ "I left Stripe, joined Notion" is the extraction prompt's own
    replacement-not-retraction example; a comma after the employer is a
    replacement-clause shape, so the employment retraction must not fire."""
    result = heuristic_extract(
        [{"role": "user", "content": "I left Stripe, joined Notion."}],
        [{"id": "m1", "type": "fact", "key": "employment.employer",
          "value": "Works at Stripe"}],
        TS,
    )
    assert not any("no longer" in m.value.lower() for m in result.memories)


def test_heuristic_retraction_wins_over_positive_pattern_in_same_turn():
    """ "my cat Mochi ... gave the cat away" must yield the retraction, not a
    fresh positive fact — the retraction's match starts later in the text,
    so it wins the same-key collision (later statement = later truth)."""
    result = heuristic_extract(
        [{"role": "user",
          "content": "It was hard: my cat Mochi was family, but we gave the cat away."}],
        [], TS,
    )
    pets = [m for m in result.memories if m.key == "pets.cat.name"]
    assert len(pets) == 1
    assert "no longer" in pets[0].value.lower()


def test_heuristic_employment_retraction_only_negates_the_recorded_employer():
    """ "I left Denver." is titlecased and clause-final but names a city, not
    the employer on record — the degraded mode must only negate what it can
    match against an existing memory, and must not negate anything when no
    employer is recorded at all."""
    existing = [{"id": "mem_3", "type": "fact", "key": "employment.employer",
                 "value": "Works at Stripe"}]
    result = heuristic_extract(
        [{"role": "user", "content": "I left Denver."}], existing, TS)
    assert not any("no longer" in m.value.lower() for m in result.memories)

    result = heuristic_extract(
        [{"role": "user", "content": "I left Stripe."}], [], TS)
    assert not any("no longer" in m.value.lower() for m in result.memories)


def test_heuristic_employment_retraction_requires_clause_end():
    # Mid-clause "left X to/for ..." implies a transition, not a retraction —
    # only the LLM path can resolve where the user landed. The heuristic must
    # stay silent rather than assert a wrong negative.
    result = heuristic_extract(
        [{"role": "user", "content": "I left Stripe to lead a new team elsewhere"}],
        [{"id": "mem_3", "type": "fact", "key": "employment.employer",
          "value": "Works at Stripe"}], TS,
    )
    assert not any("no longer" in m.value.lower() for m in result.memories)

    result = heuristic_extract(
        [{"role": "user", "content": "Big change: I left Stripe."}],
        [{"id": "mem_3", "type": "fact", "key": "employment.employer",
          "value": "Works at Stripe"}], TS,
    )
    ops = {m.key: m for m in result.memories}
    assert ops["employment.employer"].action == "supersede"
    assert "no longer works at stripe" in ops["employment.employer"].value.lower()


def test_retraction_end_to_end_chain_and_recall(client):
    client.post("/turns", json=make_turn(
        text="Say hi to my dog Biscuit, he's a handful.", timestamp="2025-01-10T09:00:00Z"))
    client.post("/turns", json=make_turn(
        text="We gave the dog away yesterday. The new apartment doesn't allow pets.",
        timestamp="2025-02-20T14:00:00Z"))

    memories = client.get("/users/u1/memories").json()["memories"]
    pets = [m for m in memories if m["key"] == "pets.dog.name"]
    active = [m for m in pets if m["active"]]
    inactive = [m for m in pets if not m["active"]]
    assert len(active) == 1 and len(inactive) == 1
    assert "no longer" in active[0]["value"].lower()
    assert "Biscuit" in inactive[0]["value"]
    assert active[0]["supersedes"] == inactive[0]["id"]

    r = client.post("/recall", json={"query": "Does the user still have a dog?",
                                     "session_id": "probe", "user_id": "u1",
                                     "max_tokens": 512})
    ctx = r.json()["context"]
    assert "no longer" in ctx.lower()
    assert "Biscuit" in ctx  # the chain renders as history, not as current truth


def test_history_decoration_falls_back_to_bare_line_under_pressure(client):
    """Review finding: near budget saturation, a decorated line must degrade
    to its bare form rather than evict the fact. Build four chained facts and
    set the budget so exactly three fit decorated and the fourth fits only
    bare — the fourth fact must appear, undecorated."""
    from memory_service import assembly, store
    from memory_service.tokens import approx_tokens

    owner = "u-pressure"
    keys = [f"{c}.fact" for c in "abcd"]
    for i, key in enumerate(keys):
        prior_id = store.insert_memory(
            owner=owner, user_id=owner, type_="fact", key=key,
            value=f"Old value {chr(65 + i)} " + "x" * 120,
            confidence=0.9, entities=[], source_session="s1", source_turn=None)
        store.insert_memory(
            owner=owner, user_id=owner, type_="fact", key=key,
            value=f"New value {chr(65 + i)} " + "y" * 120,
            confidence=0.9, entities=[], source_session="s1", source_turn=None,
            supersedes_id=prior_id)

    actives = [m for m in store.get_memories(owner, active_only=True)]
    actives.sort(key=lambda m: m["key"])
    dec_costs = [approx_tokens(assembly._memory_line(m, history_hops=1)) + 1 for m in actives]
    bare_cost = approx_tokens(assembly._memory_line(actives[3], history_hops=0)) + 1
    header_cost = approx_tokens("## Known facts about this user") + 1
    budget = header_cost + sum(dec_costs[:3]) + bare_cost
    assert 256 <= budget < 512, budget  # the hops=1 band, by construction

    context, _ = assembly.assemble(
        owner=owner, query="", retrieved={"memories": [], "turns": []},
        max_tokens=budget)
    lines = [ln for ln in context.splitlines() if ln.startswith("- ")]
    assert len(lines) == 4, context  # the fourth fact survived the wall
    assert sum("previously:" in ln for ln in lines) == 3, context
    assert "New value D" in context and "previously:" not in lines[3], context


def test_history_renders_trajectory_at_generous_budgets(client):
    """A 3-step chain (Stripe -> Notion -> Figma) renders two hops of history
    at budgets >= 512, one hop in [256, 512), none below 256."""
    client.post("/turns", json=make_turn(
        text="I work at Stripe.", timestamp="2025-01-01T09:00:00Z"))
    client.post("/turns", json=make_turn(
        text="I just started at Notion this week.", timestamp="2025-03-01T09:00:00Z"))
    client.post("/turns", json=make_turn(
        text="I am working at Figma now.", timestamp="2025-06-01T09:00:00Z"))

    def recall_ctx(budget: int) -> str:
        r = client.post("/recall", json={"query": "Where does the user work?",
                                         "session_id": "probe", "user_id": "u1",
                                         "max_tokens": budget})
        assert r.status_code == 200
        return r.json()["context"]

    generous = recall_ctx(1024)
    fact_line = next(line for line in generous.splitlines()
                     if line.startswith("- ") and "Figma" in line)
    assert "previously: Works at Notion" in fact_line
    assert "earlier: Works at Stripe" in fact_line

    medium = recall_ctx(300)
    assert "previously:" in medium
    assert "earlier:" not in medium

    tight = recall_ctx(128)
    assert "previously:" not in tight and "earlier:" not in tight
