"""Turn -> structured memories.

Primary path: Claude with structured output. The extractor sees the user's
existing ACTIVE memories and returns explicit actions:

  create     — genuinely new knowledge
  supersede  — contradicts/updates an existing memory (supersedes_id required);
               the old memory is kept inactive with a chain pointer. Retraction
               ("we gave the dog away") is supersession by the negated current
               state ("No longer has a dog") — never bare deactivation, so
               recall can distinguish "no longer has" from "never had"
  reinforce  — restates an existing memory (bumps updated_at/confidence)

Making the LLM decide supersession *with the existing memories in context* is
the core design choice: contradiction detection is a semantic problem
("I just started at Notion" vs employment.employer=Stripe), not a string
problem, and the model is the only component that can see both sides at once.

Fallback path (no API key / API down): a small rule-based extractor so the
service keeps functioning, with reduced extraction quality. Extraction can
never fail /turns: any extractor error degrades to the next tier (LLM ->
heuristic -> summary-only), and the turn plus whatever was extracted are then
persisted together in one transaction (store.write_turn).
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import config

log = logging.getLogger("memory.extraction")

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _extraction_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="extract")
    return _pool

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
13. RETRACTION (cessation without replacement): when the user states a fact has CEASED and \
nothing replaces it ("we gave Biscuit away", "I'm not vegetarian anymore", "I left Stripe" \
with no new employer), use action="supersede" on that memory with the value phrased as the \
negated CURRENT state ("No longer has a dog", "Is not vegetarian anymore", "Not currently \
employed"). Never leave the stale fact active, and never phrase the value as the transition \
event ("Gave the dog away"). When a replacement IS stated ("left Stripe, joined Notion"), \
emit only the replacement supersession — no separate retraction.

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


def _llm_extract_inner(
    turn_text: str,
    existing: list[dict[str, Any]],
    timestamp: str,
) -> ExtractionResult | None:
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


def llm_extract(
    turn_text: str,
    existing: list[dict[str, Any]],
    timestamp: str,
) -> ExtractionResult | None:
    """Returns None on any API failure or deadline; caller falls back.

    The whole call runs under a hard wall-clock deadline
    (EXTRACTION_DEADLINE_S, default 40s): the eval gives /turns 60 seconds
    total, and SDK-level retries/timeouts compose multiplicatively (and a
    429's retry-after sleep can exceed the per-attempt timeout entirely), so
    a request-level deadline is the only bound that actually holds.
    """
    if not config.ANTHROPIC_API_KEY:
        return None
    try:
        future = _extraction_pool().submit(_llm_extract_inner, turn_text, existing, timestamp)
        return future.result(timeout=config.EXTRACTION_DEADLINE_S)
    except FuturesTimeout:
        log.error("LLM extraction exceeded %.0fs deadline; falling back to heuristics",
                  config.EXTRACTION_DEADLINE_S)
        return None
    except Exception:
        log.exception("LLM extraction failed; falling back to heuristics")
        return None


# ---------------- heuristic fallback ----------------

_HEURISTICS: list[tuple[re.Pattern[str], MemoryType, str, str]] = [
    # Retraction patterns. Values are negated CURRENT state, not the
    # transition — recall must let a frozen LLM distinguish "never had" from
    # "no longer has" (README §5). Each is anchored so it cannot fire on
    # idioms or adjacent topics: "I don't have a dog in this fight" (no
    # clause end after the species), "I don't have my dog with me today"
    # ("my" excluded — that's absence, not cessation), "we gave the dog food
    # away" (the name slot requires a capitalized token), "I'm not a
    # vegetarian-hater" (bare hyphen is not a clause end). When a retraction
    # and a positive pattern hit the same key in one turn, text position
    # decides — see the collision rule in heuristic_extract.
    (re.compile(r"\b(?:I|we) (?:no longer have|don't have|do not have) (?:a |the )?(dog|cat|bird|rabbit)\b(?:\s+any ?more|\s+now)?\s*(?:[,.!;]|$)", re.I), "fact", "pets.{0}.name", "No longer has a {0}"),
    (re.compile(r"\b(?:I|[Ww]e) (?:had to )?(?:gave|give) (?:my|our|the) (dog|cat|bird|rabbit)(?: [A-Z]\w+)? away"), "fact", "pets.{0}.name", "No longer has a {0}"),
    (re.compile(r"\bI(?:'m| am) (?:not|no longer) (?:a )?(vegetarian|vegan)(?: any ?more)?\s*(?:[,.!;:—–]|$)", re.I), "preference", "diet.style", "No longer {0}"),
    # Tight terminator, no comma: "I left Stripe." retracts; "I left Stripe
    # to join Notion" and "I left Stripe, joined Notion" must not fire (the
    # replacement is the LLM path's job — prompt rule 13).
    (re.compile(r"\bI (?:quit|left) (?:my job at )?([A-Z][\w&-]*(?:\.[\w&-]+)*)\s*(?:[.!]|$)"), "fact", "employment.employer", "No longer works at {0}"),
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
    messages: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    timestamp: str,
) -> ExtractionResult:
    """Rule-based degraded mode. Supersession = exact key collision.

    Works from the structured messages, never by re-parsing rendered text —
    message content legally contains anything (newlines, lines starting with
    "user", no colons), and a fallback extractor that can crash on valid
    input is worse than no fallback (it used to: see CHANGELOG v0.7).
    """
    by_key = {m["key"]: m for m in existing}
    user_texts = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
    user_text = "\n".join(user_texts) or "\n".join(
        str(m.get("content", "")) for m in messages
    )

    # Same-key collision rule: when two patterns hit the same key in one
    # turn ("I left Stripe. I joined Notion." — retraction + replacement),
    # the match that starts LATER in the user's text wins: the later
    # statement is the later truth. Pattern-list order deciding this was a
    # v0.8 review finding — it silently dropped stated replacements.
    best_by_key: dict[str, tuple[int, ExtractedMemory]] = {}
    for pattern, type_, key_tpl, value_tpl in _HEURISTICS:
        m = pattern.search(user_text)
        if not m:
            continue
        groups = [g.strip() if g else "" for g in m.groups()]
        key = key_tpl.format(*[g.lower() for g in groups])
        value = value_tpl.format(*groups)
        prior = by_key.get(key)
        # Employment retraction is the one pattern whose captured token could
        # be anything titlecased ("I left Denver." — a city, not a job): only
        # negate an employer we actually have on record, and only when the
        # named thing matches it.
        if value_tpl.startswith("No longer works at") and (
            prior is None or groups[0].lower() not in prior["value"].lower()
        ):
            continue
        if prior is not None and prior["value"].strip().lower() == value.strip().lower():
            em = ExtractedMemory(
                action="reinforce", type=type_, key=key, value=value,
                confidence=0.6, entities=[g.lower() for g in groups if g],
                supersedes_id=prior["id"],
            )
        else:
            em = ExtractedMemory(
                action="supersede" if prior is not None else "create",
                type=type_, key=key, value=value, confidence=0.6,
                entities=[g.lower() for g in groups if g],
                supersedes_id=prior["id"] if prior is not None else None,
            )
        held = best_by_key.get(key)
        if held is None or m.start() > held[0]:
            best_by_key[key] = (m.start(), em)
    out = [em for _, em in best_by_key.values()]

    first_user = (user_texts[0] if user_texts else user_text).strip()
    summary = f"User discussed: {first_user[:200]}" if first_user else "Turn with no user text."
    return ExtractionResult(turn_summary=summary, memories=out)


def extract(
    messages: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    timestamp: str,
) -> tuple[ExtractionResult, str]:
    """Returns (result, mode) where mode is 'llm', 'heuristic', or 'minimal'.

    Never raises: any extractor failure degrades one tier further. The
    'minimal' tier stores the turn with a bare summary and no memories —
    the turn itself must never be lost to an extraction bug.
    """
    result = llm_extract(render_turn(messages), existing, timestamp)
    if result is not None:
        return result, "llm"
    try:
        return heuristic_extract(messages, existing, timestamp), "heuristic"
    except Exception:
        log.exception("heuristic extraction failed; storing turn with minimal summary")
        first = next((str(m.get("content", ""))[:200] for m in messages
                      if m.get("role") == "user"), "")
        return ExtractionResult(
            turn_summary=f"User discussed: {first}" if first else "Conversation turn.",
            memories=[],
        ), "minimal"
