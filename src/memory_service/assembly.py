"""Context assembly under a token budget.

Tier order (= drop order under pressure, last tier dropped first):

  A. "## Known facts about this user"      — active facts/preferences/opinions
  B. "## Relevant memories"                — events + dated, query-relevant items
  C. "## Relevant from recent conversations" — turn summaries

Rationale (defended in README): stable user facts are the highest-value tokens
per character for a frozen LLM — they are almost always load-bearing for
personalization and they are small. Query-relevant memories come next; raw
conversational context is the most redundant and gets whatever budget remains.
Inside each tier, lines are ranked by retrieval score (query relevance), then
recency; under pressure we cut whole lines from the bottom, never mid-line.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from . import store
from .models import Citation
from .tokens import approx_tokens

log = logging.getLogger("memory.assembly")

_TYPE_ORDER = {"fact": 0, "preference": 1, "opinion": 2}


def _date_of(ts: str | None) -> str:
    return (ts or "")[:10]


def turn_snippet(turn: dict[str, Any], max_chars: int = 240) -> str:
    if turn.get("summary"):
        return turn["summary"][:max_chars]
    try:
        messages = json.loads(turn["messages_json"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return ""
    user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
    text = " ".join(user_texts) or " ".join(str(m.get("content", "")) for m in messages)
    return text[:max_chars]


def _memory_line(mem: dict[str, Any]) -> str:
    line = f"- {mem['value']} (updated {_date_of(mem['updated_at'])}"
    if mem.get("supersedes"):
        prior = store.get_memory(mem["supersedes"])
        if prior:
            line += f"; previously: {prior['value'][:120]}"
    return line + ")"


def _event_line(mem: dict[str, Any]) -> str:
    return f"- [{_date_of(mem['created_at'])}] {mem['value']}"


class _Builder:
    def __init__(self, max_tokens: int):
        self.budget = max_tokens
        self.lines: list[str] = []
        self.citations: list[Citation] = []

    def try_add(self, line: str, citation: Citation | None = None) -> bool:
        cost = approx_tokens(line) + 1  # +1 for the newline
        if cost > self.budget:
            return False
        self.budget -= cost
        self.lines.append(line)
        if citation is not None:
            self.citations.append(citation)
        return True


def _mem_citation(mem: dict[str, Any], score: float) -> Citation | None:
    turn_id = mem.get("source_turn")
    if not turn_id:
        return None
    return Citation(turn_id=turn_id, score=round(score, 4), snippet=mem["value"][:160])


def assemble(
    *,
    owner: str,
    query: str,
    retrieved: dict[str, list[tuple[str, float]]],
    max_tokens: int,
) -> tuple[str, list[Citation]]:
    mem_scores = dict(retrieved.get("memories", []))
    turn_scores = dict(retrieved.get("turns", []))

    actives = store.get_memories(owner, active_only=True)
    profile = [m for m in actives if m["type"] in _TYPE_ORDER]
    events = [m for m in actives if m["type"] == "event"]

    # Nothing known and nothing matched: cold/noise -> empty context, never an error.
    if not profile and not events and not turn_scores:
        return "", []

    def profile_rank(m: dict[str, Any]) -> tuple:
        return (
            _TYPE_ORDER[m["type"]],
            -mem_scores.get(m["id"], 0.0),
            -(m["confidence"] or 0),
            m["key"],
        )

    profile.sort(key=profile_rank)
    events.sort(key=lambda m: -mem_scores.get(m["id"], 0.0))

    b = _Builder(max_tokens)

    if profile:
        b.try_add("## Known facts about this user")
        for mem in profile:
            if not b.try_add(_memory_line(mem), _mem_citation(mem, mem_scores.get(mem["id"], 0.0))):
                break

    relevant_events = [m for m in events]
    if relevant_events:
        b.try_add("\n## Relevant memories")
        for mem in relevant_events:
            if not b.try_add(_event_line(mem), _mem_citation(mem, mem_scores.get(mem["id"], 0.0))):
                break

    ranked_turns = sorted(turn_scores.items(), key=lambda x: -x[1])
    if ranked_turns:
        b.try_add("\n## Relevant from recent conversations")
        for turn_id, score in ranked_turns:
            turn = store.get_turn(turn_id)
            if not turn:
                continue
            snippet = turn_snippet(turn)
            line = f"- [{_date_of(turn['ts'])}] {snippet}"
            if not b.try_add(line, Citation(turn_id=turn_id, score=round(score, 4), snippet=snippet[:160])):
                break

    context = "\n".join(b.lines).strip()
    if not context.strip("#\n "):
        return "", []
    return context, b.citations
