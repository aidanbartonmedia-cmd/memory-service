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


_HISTORY_LABELS = ("previously", "earlier")


def _memory_line(mem: dict[str, Any], *, history_hops: int = 1) -> str:
    """Render a fact with up to `history_hops` of its supersession chain
    inline: "X (updated D; previously: Y; earlier: Z)". Two hops is enough to
    show an opinion arc's trajectory (settled stance <- middle <- origin)
    without letting a long chain eat the budget that current facts need."""
    line = f"- {mem['value'][:300]} (updated {_date_of(mem['updated_at'])}"
    cur = mem
    for hop in range(min(history_hops, len(_HISTORY_LABELS))):
        if not cur.get("supersedes"):
            break
        prior = store.get_memory(cur["supersedes"])
        if not prior:
            break
        line += f"; {_HISTORY_LABELS[hop]}: {prior['value'][:120]}"
        cur = prior
    return line + ")"


def _event_line(mem: dict[str, Any]) -> str:
    return f"- [{_date_of(mem['created_at'])}] {mem['value']}"


_MAX_CITATIONS = 12
# Below this budget we drop "previously: ..." decorations — history is the
# first nicety to cut when every token competes with a current fact. At
# generous budgets we render a second hop ("earlier: ...") so a 3-step
# opinion arc reads as a trajectory, not a single before/after.
_HISTORY_MIN_BUDGET = 256
_TRAJECTORY_MIN_BUDGET = 512


def _mem_citation(mem: dict[str, Any], score: float) -> Citation | None:
    turn_id = mem.get("source_turn")
    if not turn_id:
        return None
    return Citation(turn_id=turn_id, score=round(score, 4), snippet=mem["value"][:160])


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    best: dict[str, Citation] = {}
    for c in citations:
        prev = best.get(c.turn_id)
        if prev is None or c.score > prev.score:
            best[c.turn_id] = c
    return sorted(best.values(), key=lambda c: -c.score)[:_MAX_CITATIONS]


def assemble(
    *,
    owner: str,
    query: str,
    retrieved: dict[str, Any],
    max_tokens: int,
) -> tuple[str, list[Citation]]:
    """Render tiers A/B/C greedily under the budget.

    Sections are built as (header, items) and a header is only emitted when
    at least one of its items fits — no orphan headers at tiny budgets.
    """
    mem_scores = dict(retrieved.get("memories", []))
    turn_scores = dict(retrieved.get("turns", []))
    if max_tokens >= _TRAJECTORY_MIN_BUDGET:
        history_hops = 2
    elif max_tokens >= _HISTORY_MIN_BUDGET:
        history_hops = 1
    else:
        history_hops = 0

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

    # Items are (line, bare_fallback, citation). The fallback is the same
    # fact without history decorations: a "previously: ..." hop must never
    # cost the fact itself its place in the context (review finding — near
    # budget saturation, decorated lines could evict lower-ranked facts that
    # their bare forms would have fit).
    sections: list[tuple[str, list[tuple[str, str | None, Citation | None]]]] = []

    profile_items = []
    for mem in profile:
        line = _memory_line(mem, history_hops=history_hops)
        bare = _memory_line(mem, history_hops=0) if history_hops else None
        profile_items.append((
            line, bare if bare != line else None,
            _mem_citation(mem, mem_scores.get(mem["id"], 0.0)),
        ))
    if profile_items:
        sections.append(("## Known facts about this user", profile_items))

    # Tier B is "query-relevant memories" (the spec's words): only events the
    # retrieval layer actually surfaced for THIS query, best-first, capped —
    # not a dump of every event the user ever mentioned.
    scored_events = [m for m in events if mem_scores.get(m["id"], 0.0) > 0.0][:5]
    event_items = [
        (_event_line(mem), None, _mem_citation(mem, mem_scores.get(mem["id"], 0.0)))
        for mem in scored_events
    ]
    if event_items:
        sections.append(("## Relevant memories", event_items))

    turn_items: list[tuple[str, str | None, Citation | None]] = []
    for turn_id, score in sorted(turn_scores.items(), key=lambda x: -x[1]):
        turn = store.get_turn(turn_id)
        if not turn:
            continue
        snippet = turn_snippet(turn)
        turn_items.append((
            f"- [{_date_of(turn['ts'])}] {snippet}", None,
            Citation(turn_id=turn_id, score=round(score, 4), snippet=snippet[:160]),
        ))
    if turn_items:
        sections.append(("## Relevant from recent conversations", turn_items))

    budget = max_tokens
    out_lines: list[str] = []
    citations: list[Citation] = []
    for header, items in sections:
        header_text = header if not out_lines else "\n" + header
        header_cost = approx_tokens(header_text) + 1
        if header_cost >= budget:
            break
        section_budget = budget - header_cost
        section_lines: list[str] = []
        for line, bare, citation in items:
            cost = approx_tokens(line) + 1
            if cost > section_budget:
                # History decorations are a nicety; the fact is not. Fall
                # back to the undecorated line before giving up on it.
                if bare is not None:
                    cost = approx_tokens(bare) + 1
                    if cost <= section_budget:
                        section_budget -= cost
                        section_lines.append(bare)
                        if citation is not None:
                            citations.append(citation)
                        continue
                break  # items are ranked; everything after is lower priority
            section_budget -= cost
            section_lines.append(line)
            if citation is not None:
                citations.append(citation)
        if section_lines:
            out_lines.append(header_text)
            out_lines.extend(section_lines)
            budget = section_budget
        # If nothing fit, the header was never charged; try the next tier
        # (its lines may be shorter).

    context = "\n".join(out_lines).strip()

    # Tiny-budget fallback: when even "header + one line" doesn't fit, emit
    # the single top-priority line bare rather than nothing — at
    # max_tokens=16 the most load-bearing known fact beats an empty context.
    if not context.strip("#\n ") and sections:
        for _header, items in sections:
            for line, bare, citation in items:
                candidate = bare if bare is not None else line
                if approx_tokens(candidate) <= max_tokens:
                    return candidate, ([citation] if citation else [])
            break  # only the top tier; lower tiers are lower priority
        return "", []

    if not context.strip("#\n "):
        return "", []
    return context, _dedupe_citations(citations)
