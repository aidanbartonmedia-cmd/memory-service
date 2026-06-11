"""Context assembly under a token budget. v0.1 — minimal turn-snippet context."""

from __future__ import annotations

import json
import logging
from typing import Any

from . import store
from .models import Citation
from .tokens import approx_tokens

log = logging.getLogger("memory.assembly")


def _date_of(ts: str) -> str:
    return (ts or "")[:10]


def turn_snippet(turn: dict[str, Any], max_chars: int = 240) -> str:
    if turn.get("summary"):
        return turn["summary"][:max_chars]
    try:
        messages = json.loads(turn["messages_json"])
    except (KeyError, json.JSONDecodeError):
        return ""
    user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
    text = " ".join(user_texts) or " ".join(m.get("content", "") for m in messages)
    return text[:max_chars]


def assemble(
    *,
    owner: str,
    retrieved: dict[str, list[tuple[str, float]]],
    max_tokens: int,
) -> tuple[str, list[Citation]]:
    """v0.1: render matched turns as dated lines, greedy under budget."""
    lines: list[str] = []
    citations: list[Citation] = []
    budget = max_tokens

    header = "## Relevant from recent conversations"
    if retrieved["turns"]:
        budget -= approx_tokens(header)
        lines.append(header)

    for turn_id, score in retrieved["turns"]:
        turn = store.get_turn(turn_id)
        if not turn:
            continue
        snippet = turn_snippet(turn)
        line = f"- [{_date_of(turn['ts'])}] {snippet}"
        cost = approx_tokens(line)
        if cost > budget:
            break
        budget -= cost
        lines.append(line)
        citations.append(Citation(turn_id=turn_id, score=round(score, 4), snippet=snippet[:160]))

    if not citations:
        return "", []
    return "\n".join(lines), citations
