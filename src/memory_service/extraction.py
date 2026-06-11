"""Turn -> structured memories.

Primary path: Claude with structured output. The extractor sees the user's
existing ACTIVE memories and returns explicit actions:

  create     — genuinely new knowledge
  supersede  — contradicts/updates an existing memory (supersedes_id required);
               the old memory is kept inactive with a chain pointer
  reinforce  — restates an existing memory (bumps updated_at/confidence)

Making the LLM decide supersession *with the existing memories in context* is
the core design choice: contradiction detection is a semantic problem
("I just started at Notion" vs employment.employer=Stripe), not a string
problem, and the model is the only component that can see both sides at once.

Fallback path (no API key / API down): a small rule-based extractor so the
service keeps functioning, with reduced extraction quality. /turns never
fails because extraction failed — the raw turn is always stored first.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import config

log = logging.getLogger("memory.extraction")

MemoryType = Literal["fact", "preference", "opinion", "event"]


class ExtractedMemory(BaseModel):
    action: Literal["create", "supersede", "reinforce"]
    type: MemoryType
    key: str = Field(description="canonical dotted key, e.g. employment.employer")
    value: str = Field(description="the current truth, phrased as a standalone statement")
    confidence: float = Field(ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list)
    supersedes_id: str | None = Field(
        default=None, description="id of the existing memory this replaces (supersede/reinforce)"
    )


class ExtractionResult(BaseModel):
    turn_summary: str = Field(description="one dated third-person sentence about this turn")
    memories: list[ExtractedMemory] = Field(default_factory=list)


_SYSTEM = """You are the extraction stage of a long-term memory service for an AI assistant. \
You receive one conversation turn plus the user's existing active memories, and you produce \
structured memory operations.

WHAT TO EXTRACT (about the USER only):
- facts: employment, location, family, pets, health constraints (allergies), possessions
- preferences: communication style, food, tools, likes/dislikes that guide future behavior
- opinions: stances that can evolve ("loves TypeScript") — capture the CURRENT stance
- events: time-bound items (an interview next week, a recital, a move in progress)

RULES:
1. Only extract what the USER asserts or clearly implies. The assistant's statements are not \
user facts. Tool results are not user facts unless the user confirms them.
2. Capture implicit facts: "walking Biscuit this morning" implies the user has a pet named \
Biscuit (pets.dog.name or pets.pet.name if species unknown).
3. Corrections inside the turn ("actually, not peanuts — shellfish") must yield ONLY the \
corrected fact. Never emit the misspoken version.
4. SUPERSESSION: if a new statement contradicts or updates an existing active memory listed \
below (same topic: job, city, allergy, stance...), use action="supersede" with that memory's \
id. The value is the new current truth. Do NOT use create for a changed fact.
5. REINFORCE: if the turn merely restates an existing memory, use action="reinforce" with its \
id (no duplicate creation).
6. Opinion arcs: when a stance shifts ("love TS" -> "TS is fine for big projects, Python for \
scripts"), supersede the old opinion; the new value should describe the current nuanced \
stance on its own.
7. keys are lowercase dotted paths under these prefixes when applicable: employment.*, \
home.*, family.*, pets.*, diet.*, health.*, preferences.*, opinions.*, events.*, projects.*. \
Reuse the existing memory's key when superseding/reinforcing.
8. value must stand alone without the conversation ("Works at Notion as a product manager", \
not "started there this week"). Phrase fact values as the CURRENT state of the world — \
"Lives in Denver", "Works at Notion" — never as the transition that produced the state \
("Relocating to Denver", "Just started at Notion"). Recall queries ask about the state \
("where does the user live?"), so state phrasing is what retrieval must match. The \
transition itself, if notable, is a separate event memory.
9. entities: short lowercase tags for every salient proper noun AND category in the memory \
(["notion", "product manager", "employment"], ["biscuit", "dog", "pet"]). These link \
memories for multi-hop recall — be generous and consistent.
10. confidence: 0.95+ explicit statement, 0.7-0.9 strong implication, <0.7 speculative.
11. Do not extract trivia about the current task itself (code being debugged is context, not \
a durable fact) — but DO extract it as part of turn_summary.
12. turn_summary: ONE third-person sentence, starting "User ...", with the salient specifics \
(names, places, topics). It is a retrieval target — include keywords.

Return zero memories when the turn contains nothing durable about the user."""


def render_turn(messages: list[dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user")
        name = f" ({m['name']})" if m.get("name") else ""
        content = str(m.get("content", ""))[:4000]
        lines.append(f"{role}{name}: {content}")
    return "\n".join(lines)


def render_existing(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "(none yet)"
    lines = []
    for m in memories:
        lines.append(f"- id={m['id']} [{m['type']}] {m['key']} = {m['value']}")
    return "\n".join(lines)


def llm_extract(
    turn_text: str,
    existing: list[dict[str, Any]],
    timestamp: str,
) -> ExtractionResult | None:
    """Returns None on any API failure; caller falls back to heuristics."""
    if not config.ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(
            max_retries=config.LLM_MAX_RETRIES, timeout=config.LLM_TIMEOUT_S
        )
        prompt = (
            f"EXISTING ACTIVE MEMORIES for this user:\n{render_existing(existing)}\n\n"
            f"TURN (timestamp {timestamp}):\n{turn_text}"
        )
        response = client.messages.parse(
            model=config.LLM_MODEL,
            max_tokens=4000,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=ExtractionResult,
        )
        result = response.parsed_output
        if result is None:
            log.warning("extraction parse returned no output (stop_reason=%s)", response.stop_reason)
        return result
    except Exception:
        log.exception("LLM extraction failed; falling back to heuristics")
        return None


# ---------------- heuristic fallback ----------------

_HEURISTICS: list[tuple[re.Pattern[str], MemoryType, str, str]] = [
    (re.compile(r"\bI(?:'m| am)? (?:work(?:ing)? at|employed (?:at|by)) ([A-Z][\w&.-]*)"), "fact", "employment.employer", "Works at {0}"),
    (re.compile(r"\bI (?:just )?(?:started|joined) (?:at |working at )?([A-Z][\w&.-]*)"), "fact", "employment.employer", "Works at {0}"),
    (re.compile(r"\bI (?:live|stay) in ([A-Z][\w .-]*?)(?:[,.!]|$)"), "fact", "home.city", "Lives in {0}"),
    (re.compile(r"\b(?:I |we )?(?:just )?moved to ([A-Z][\w .-]*?)(?:\s+from\s+[A-Z][\w .-]*)?(?:[,.!]| last| this|$)"), "fact", "home.city", "Lives in {0} (recently moved)"),
    (re.compile(r"\ballergic to (\w+(?: \w+)?)", re.I), "fact", "diet.allergy", "Allergic to {0}"),
    (re.compile(r"\bmy (dog|cat|bird|rabbit)(?: is)?(?:,| named| called)? ([A-Z]\w+)"), "fact", "pets.{0}.name", "Has a {0} named {1}"),
    (re.compile(r"\b([A-Z]\w+) is my (dog|cat|bird|rabbit)"), "fact", "pets.{1}.name", "Has a {1} named {0}"),
    (re.compile(r"\bmy favorite (\w+(?: \w+)?) is (\w+(?: \w+)?)", re.I), "preference", "preferences.{0}", "Favorite {0}: {1}"),
    (re.compile(r"\b(\w+(?: \w+)?) is (?:actually )?my favorite", re.I), "preference", "preferences.general", "Favorite: {0}"),
    (re.compile(r"\bI(?: really)? prefer ([^,.!]{3,60})", re.I), "preference", "preferences.general", "Prefers {0}"),
    (re.compile(r"\bI(?:'m| am) a(?:n)? ([a-z]+(?: [a-z]+)?) (?:engineer|developer|designer|photographer|manager|writer)", re.I), "fact", "employment.role", "Is a {0} professional"),
    (re.compile(r"\bI(?:'m| am) vegetarian", re.I), "preference", "diet.style", "Is vegetarian"),
    (re.compile(r"\bI(?:'m| am) vegan", re.I), "preference", "diet.style", "Is vegan"),
]


def heuristic_extract(
    turn_text: str,
    existing: list[dict[str, Any]],
    timestamp: str,
) -> ExtractionResult:
    """Rule-based degraded mode. Supersession = exact key collision."""
    by_key = {m["key"]: m for m in existing}
    user_text = "\n".join(
        line.split(":", 1)[1] for line in turn_text.splitlines()
        if line.startswith("user")
    ) or turn_text

    out: list[ExtractedMemory] = []
    seen_keys: set[str] = set()
    for pattern, type_, key_tpl, value_tpl in _HEURISTICS:
        m = pattern.search(user_text)
        if not m:
            continue
        groups = [g.strip() if g else "" for g in m.groups()]
        key = key_tpl.format(*[g.lower() for g in groups])
        value = value_tpl.format(*groups)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        prior = by_key.get(key)
        if prior is not None and prior["value"].strip().lower() == value.strip().lower():
            out.append(ExtractedMemory(
                action="reinforce", type=type_, key=key, value=value,
                confidence=0.6, entities=[g.lower() for g in groups if g],
                supersedes_id=prior["id"],
            ))
        else:
            out.append(ExtractedMemory(
                action="supersede" if prior is not None else "create",
                type=type_, key=key, value=value, confidence=0.6,
                entities=[g.lower() for g in groups if g],
                supersedes_id=prior["id"] if prior is not None else None,
            ))

    first_user = next((line for line in turn_text.splitlines() if line.startswith("user")), turn_text)
    summary = f"User discussed: {first_user.split(':', 1)[-1].strip()[:200]}"
    return ExtractionResult(turn_summary=summary, memories=out)


def extract(
    messages: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    timestamp: str,
) -> tuple[ExtractionResult, str]:
    """Returns (result, mode) where mode is 'llm' or 'heuristic'."""
    turn_text = render_turn(messages)
    result = llm_extract(turn_text, existing, timestamp)
    if result is not None:
        return result, "llm"
    return heuristic_extract(turn_text, existing, timestamp), "heuristic"
