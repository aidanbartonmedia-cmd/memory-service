"""Recall-quality fixture test (spec §7: required).

Ingests the scripted conversations from fixtures/ and runs every probe
against /recall, reporting "X of Y expected facts appeared in context".

By default this runs in heuristic-extraction mode (no API key, fast,
deterministic) and asserts a degraded-mode floor — the heuristic extractor
catches pattern-shaped facts (employment, location, pets, allergies) but
not implicit facts, corrections, or opinion arcs, so the bar is deliberately
lower. The full-quality loop (LLM extraction, 37/37 expected) is
scripts/selfeval.py against a running service; CHANGELOG quotes those runs.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "conversations.json"

# Probes a regex extractor cannot pass (implicit facts, opinion arcs,
# corrections, paraphrases that need dense matching on LLM-phrased values,
# name-based retractions like "gave Mochi away" that need entity resolution).
# They run anyway (must not crash) but are excluded from the floor.
HEURISTIC_EXEMPT_CATEGORIES = {"implicit", "opinion_arc", "correction", "paraphrase",
                               "multi_hop", "keyword", "retraction"}


def test_recall_quality_fixture(client):
    fixture = json.loads(FIXTURE.read_text())

    for convo in fixture["conversations"]:
        for session in convo["sessions"]:
            for turn in session["turns"]:
                r = client.post("/turns", json={
                    "session_id": session["session_id"],
                    "user_id": convo["user_id"],
                    "messages": turn["messages"],
                    "timestamp": turn["timestamp"],
                    "metadata": {},
                })
                assert r.status_code == 201

    scored = 0
    passed = 0
    results = []
    for probe in fixture["probes"]:
        r = client.post("/recall", json={
            "query": probe["query"],
            "session_id": probe["session_id"],
            "user_id": probe["user_id"],
            "max_tokens": probe.get("max_tokens", 512),
        })
        assert r.status_code == 200, probe["id"]
        ctx = r.json()["context"]

        if probe.get("expect_empty"):
            ok = ctx.strip() == ""
        elif probe.get("expect_any"):
            ok = any(n.lower() in ctx.lower() for n in probe["expect_any"])
        else:
            continue

        exempt = probe["category"] in HEURISTIC_EXEMPT_CATEGORIES
        results.append((probe["id"], probe["category"], ok, exempt))
        if not exempt:
            scored += 1
            passed += bool(ok)

    print(f"\nrecall quality (heuristic extraction): {passed}/{scored} scored probes "
          f"({len(results) - scored} exempt categories also ran without error)")
    for pid, cat, ok, exempt in results:
        print(f"  {pid:5} {cat:14} {'PASS' if ok else 'fail'}{' (exempt)' if exempt else ''}")

    # Degraded-mode floor: pattern-shaped facts and noise gating must work
    # even without an LLM. (LLM mode is measured by scripts/selfeval.py.)
    assert passed / scored >= 0.65, f"heuristic-mode recall quality regressed: {passed}/{scored}"
