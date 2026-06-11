# memory-service

A memory backend for an AI agent: ingests conversation turns, extracts
structured knowledge with an LLM, and answers recall queries with hybrid
retrieval under a token budget — with fact supersession, multi-hop entity
linking, and calibrated noise resistance.

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY (recommended — see Failure modes)
docker compose up -d
until curl -sf http://localhost:8080/health; do sleep 1; done
```

Port `8080`, named volume `memory-data`, no manual setup. Auth is optional:
set `MEMORY_AUTH_TOKEN` to require `Authorization: Bearer <token>` (ignored
when unset).

---

## 1. Architecture

```
                  POST /turns                            POST /recall
                       │                                      │
                       ▼                                      ▼
         ┌─────────────────────────┐          ┌──────────────────────────────┐
         │ 1. persist raw turn     │          │ 1. hybrid retrieval          │
         │    (never lost, even if │          │    BM25 (FTS5+porter) ───┐   │
         │    extraction fails)    │          │    dense (bge-small) ────┤   │
         ├─────────────────────────┤          │      → RRF fusion        │   │
         │ 2. LLM extraction       │          │ 2. + one entity hop      │   │
         │    (Claude, structured  │          │    (shared-tag expansion)│   │
         │    output) sees existing│          │ 3. relevance gate        │   │
         │    ACTIVE memories →    │          │    (calibrated two-tier  │   │
         │    create / supersede / │          │    floor → "" on noise)  │   │
         │    reinforce            │          │ 4. tiered assembly under │   │
         ├─────────────────────────┤          │    token budget          │   │
         │ 3. embed memory + turn  │          │    facts → events → conv │   │
         │    summary (local ONNX) │          └──────────────┬───────────┘   │
         └────────────┬────────────┘                         │               │
                      ▼                                      ▼               │
         ┌──────────────────────────────────────────────────────────────────┐
         │            SQLite (WAL, synchronous=FULL) on /data volume        │
         │   turns ─ memories(+supersession chain) ─ FTS5 ─ embeddings      │
         └──────────────────────────────────────────────────────────────────┘
```

One process, one container, one store. All LLM spend happens at **write
time** (`/turns` has a 60s budget and writes are rare); **read time is
LLM-free** (recall p50 ≈ 5ms on the self-eval fixture). Everything commits
inside the request — when `/turns` returns 201, the extracted memories are
already visible to `/recall` and `/users/{id}/memories`. No queues, no
eventual consistency, by construction.

The extraction stage is the one component allowed to be smart, and it is
positioned where it can be: it sees the new turn *and* the user's existing
active memories in one context, so contradiction detection ("just started at
Notion" vs `employment.employer = Stripe`) is a reasoning decision, not a
string comparison.

## 2. Backing store: SQLite (WAL) + FTS5, embeddings as columns

| Requirement (spec) | How SQLite delivers it |
|---|---|
| Synchronous read-after-write | Single writer connection, `synchronous=FULL`; the extraction transaction commits before the 201 |
| Persistence across restarts | One file on the named volume; WAL recovers cleanly from kill -9 |
| Hybrid retrieval | FTS5 gives BM25 with porter stemming natively; 384-d vectors live in a BLOB column, ranked brute-force in numpy |
| Eval scale (few sessions, single user) | Brute-force cosine over a few hundred rows is ~0ms; an ANN index would add a service and a failure mode and win nothing |

I considered Postgres+pgvector (a second container, real value only past
~100K vectors), Qdrant (same, plus a second *kind* of store to keep
consistent with the relational one — supersession chains want transactions),
and Redis (persistence semantics are the hard part, not the easy part).
At this workload, every one of them is an extra way to violate the
synchronous-correctness constraint; SQLite makes that constraint free. The
design is deliberately not horizontally scalable — per the brief — but the
seams are clean: `retrieval.py` doesn't care where candidates come from, so
swapping FTS5/numpy for OpenSearch/pgvector is a module change, not a
rearchitecture.

## 3. Extraction pipeline

`/turns` → persist raw turn → render the turn + the user's existing active
memories (id/type/key/value) into one prompt → Claude (`claude-opus-4-8`,
structured output validated against a pydantic schema) returns:

- `turn_summary` — one dated third-person sentence, embedded and indexed as a
  retrieval target for the conversation tier,
- a list of memory operations, each: `action` (`create` / `supersede` /
  `reinforce`), `type` (`fact|preference|opinion|event`), canonical dotted
  `key` (`employment.employer`, `pets.dog.name`), standalone `value` phrased
  as **current state** ("Lives in Denver", not "Relocating to Denver" — a
  measured fix, see CHANGELOG v0.4), `confidence`, `entities` (lowercase tags
  that power multi-hop), and `supersedes_id` when replacing.

What it extracts: explicit personal facts, preferences, opinions (current
stance), time-bound events, implicit facts ("walking Biscuit this morning" →
`pets.dog.name = Biscuit`), and in-turn corrections ("actually, not peanuts —
shellfish") which yield *only* the corrected fact. Assistant statements and
tool outputs are not treated as user facts unless the user confirms them.

What it misses, knowingly: facts the model judges non-durable (the bug being
debugged is summary material, not a memory); cross-turn inference chains that
require more than the active-memory context; and anything in degraded mode
(below). Confidence encodes the explicit/implied distinction (≥0.95 explicit,
0.7–0.9 implied) so downstream consumers can filter.

Cost/latency: one extraction call per turn, ~4.3s with Opus 4.8 on the
fixture — inside the 60s `/turns` budget the spec grants, spent exactly where
the spec says to spend it (extraction quality). `MEMORY_LLM_MODEL` switches
models per environment.

## 4. Recall strategy

`/recall` end-to-end (all local, no LLM call):

1. **Two retrievers** over the user's scope: BM25 via FTS5 (porter-stemmed,
   query stripped of stopwords — including "user", which our own summaries
   would otherwise match on every document) and cosine over local embeddings
   (bge-small-en-v1.5 ONNX, asymmetric query prefix), both over memories and
   turn summaries.
2. **Reciprocal rank fusion** (k=60) — fuses rank positions, so no score
   calibration between BM25 and cosine is needed. Hybrid because each side
   provably fails alone: "what's their dog's *name*?" is keyword-anchored;
   "what part of the world is this person located in" has zero token overlap
   with "Lives in Berlin".
3. **One entity hop**: extraction-time entity tags link memories; any memory
   sharing a tag with a top-fused hit joins the ranking at `parent × 0.5`.
   This is what answers "what city does the user with the dog named Biscuit
   live in?" — the location memory shares no tokens with the query and
   arrives via the `biscuit`/`pet` link.
4. **Relevance gate** (noise resistance): relevant iff max dense similarity
   ≥ 0.62, *or* ≥ 0.50 with a stemmed-keyword confirmation. Below the gate →
   `{"context": "", "citations": []}`, 200, never an error. The two floors
   are calibrated, not guessed: on the fixture, the hardest noise probe peaks
   at 0.585 and the weakest genuine probe sits at 0.592 (calibration data and
   both gate bugs found on the way are in CHANGELOG v0.4). Gating applies to
   `/recall` only — `/search` is an explicit tool call and stays best-effort.
5. **Tiered assembly under the budget** (chars/4 ≈ tokens; contract allows 2×,
   measured usage stays ≤ 0.97×):

   - **A. Known facts about this user** — all active facts/preferences/
     opinions, ordered fact→preference→opinion, then query-relevance, then
     confidence. Stable facts first because they are the highest value-per-
     token for a frozen LLM and they're small.
   - **B. Relevant memories** — dated events.
   - **C. Relevant from recent conversations** — turn summaries by fused score.

   Priority under pressure = exactly that order (the spec's order). Cuts are
   whole-line, bottom-up within a tier; a tier header is only emitted if at
   least one line fits; "previously: …" history decorations are the first
   nicety dropped (budgets < 256). At `max_tokens=32` you get the single
   most load-bearing known fact and nothing else.

   Citations map every rendered line to its source turn (`source_turn` for
   memories, the turn itself for tier C), deduped by turn keeping max score.

**Scoping rule:** memories are keyed by `user_id` and intentionally shared
across that user's sessions (that's what makes session-2 recall of a
session-1 fact work — and the eval's own smoke test expects it). Anonymous
turns (`user_id: null`) are scoped to `session:<session_id>` and can never
bleed across sessions. A recall that passes `user_id: null` but names a
session whose turns were written under a user resolves to that user's scope —
the session demonstrably belongs to them. Concurrent users are isolated by
the same key; the test suite hammers this in parallel threads.

## 5. Fact evolution

- **Detection** is the extractor's job, with the user's active memories in
  context — same-topic recognition is semantic ("just started at Notion"
  updates `employment.employer` even though the sentence never mentions
  employment or Stripe).
- **Storage**: supersession writes a doubly-linked chain — old row keeps
  `superseded_by`, new row keeps `supersedes`, old row flips `active=0`,
  nothing is deleted. `/users/{id}/memories` exposes the full chain.
- **Recall** renders only active facts, with one hop of history inline:
  `Works at Notion as a product manager (updated 2025-04-05; previously:
  Works at Stripe as a backend engineer)`.
- **Corrections** inside a turn never materialize the wrong fact; corrections
  of an earlier turn supersede it.
- **Opinion arcs** are modeled as supersession chains over `opinions.*` keys
  where each value is the *current nuanced stance*, not a delta: "loves
  TypeScript" → superseded by → "TypeScript is right for big team projects,
  Python for scripts". The arc is preserved and inspectable; recall leads
  with the settled stance. Partial by design: a 3+ step arc renders one hop
  of history inline rather than a narrated trajectory (CHANGELOG: next).
- **Reinforcement**: restating an existing fact bumps `updated_at`/confidence
  instead of creating a duplicate.
- **Cleanup repair**: `DELETE /sessions/{id}` removes that session's memories
  and *reactivates* any memory they had superseded, so eval cleanup can't
  leave a user with no active employer because session 3 was deleted.

## 6. Tradeoffs

- **Write-time intelligence, read-time speed.** Extraction spends seconds and
  dollars per turn; recall is milliseconds and free. Right shape for an
  agent: reads dominate writes, and recall latency sits inside a user-facing
  loop. The cost is that recall can't reason — it can only retrieve what
  extraction structured. Mitigated by extraction richness (entities, state
  phrasing, summaries); accepted otherwise.
- **LLM-in-the-loop extraction** over rules/NER: the fixture's correction,
  implicit-fact, and opinion-arc probes are simply out of reach of the
  rule-based path (that's why the fallback exists *and* why it's the
  fallback). Cost: nondeterminism — mitigated by schema-validated output,
  state-phrasing rules, and two independent full-ingest stability runs at
  24/24.
- **SQLite over a vector DB**: transactional supersession + read-after-write
  beats ANN performance we don't need at this scale (§2).
- **Brute-force cosine** is O(n) per query — correct until ~10⁵ memories per
  user, which a conversational agent won't hit; the seam to swap is one module.
- **Conservative noise gate**: I tuned to prefer empty context over plausible
  junk, because a frozen LLM treats injected context as true — false context
  is worse than no context. The margins are honest-but-thin (calibrated on 24
  probes); floors are env-tunable and the calibration method is in the repo.
- **chars/4 token approximation**: dependency-free and within the contract's
  2× tolerance (measured ≤ 0.97×); exact tokenization would add a tokenizer
  dependency to save tokens nobody is losing.

## 7. Failure modes

| Condition | Behavior |
|---|---|
| No `ANTHROPIC_API_KEY` | Service runs; extraction degrades to the rule-based fallback (regex patterns for employment/location/pets/allergies/preferences, supersession by exact key collision). Recall, persistence, contract all intact. Logged at startup-adjacent call sites. |
| Anthropic API down / timeout / over quota | Same degradation, per turn: 2 retries (SDK), 45s timeout, then heuristic fallback. The raw turn was persisted *before* extraction started — `POST /turns` still returns 201 and the turn is recallable through the conversation tier. |
| Embedding model unavailable (corrupt cache; it's baked into the image so this is exotic) | Dense retrieval and the dense gate disable; BM25-only retrieval with keyword-evidence gating. Weaker paraphrase recall and noise resistance, no crash. |
| Cold store / unknown user / empty query | `200 {"context": "", "citations": []}` — never an error. |
| Malformed JSON, missing fields, wrong types, unicode oddities, FTS metacharacters | 400 (422-class) with a JSON error body; fuzz cases in the test suite. |
| Oversized payload | 413 before body parsing (default cap 8MB). |
| Kill mid-write / `docker compose down` mid-write | WAL: committed turns survive, the in-flight transaction rolls back atomically — no torn state. Verified by the restart tests. |
| Slow disk | Everything is synchronous, so slow disk = slower 201s, not inconsistency. `/recall` does one read transaction. |

## 8. How to run the tests

```bash
# Contract / robustness / persistence / scoping / auth — fast, no API key,
# no network (runs extraction in heuristic mode):
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q          # 29 tests, <1s

# Recall-quality self-eval (the iteration loop; uses the LLM extractor,
# needs ANTHROPIC_API_KEY in the service environment):
docker compose up -d
.venv/bin/python scripts/selfeval.py          # ingests fixtures/, runs 24 probes
```

`scripts/selfeval.py` prints per-category pass rates and writes a JSON report
to `selfeval-results/` — the numbers quoted in `CHANGELOG.md` are those
files, committed. Current state: **24/24** (fact evolution, multi-hop,
corrections, opinion arc, implicit facts, noise, keyword, cold-session),
stable across two independent fresh ingests, recall p50 ≈ 5ms, budget usage
≤ 0.97× at every budget from 16 to 2048 tokens.

---

### Repo map

```
src/memory_service/
  app.py         HTTP layer: 7 contract endpoints, auth, guards, error shape
  extraction.py  Claude structured extraction + heuristic fallback (the core)
  retrieval.py   BM25 + dense + RRF + entity hop + calibrated relevance gate
  assembly.py    tiered, budget-safe context building + citations
  store.py       turns/memories CRUD, supersession chains, delete-repair
  db.py          SQLite WAL schema + serialized transactions
  embeddings.py  local ONNX embeddings (bge-small), degradation handling
  models.py      lenient-in / exact-out pydantic shapes
  config.py      env knobs (all defaulted)
tests/           contract, robustness, persistence, scoping, auth
fixtures/        5 scripted conversations + 24 probes (the self-eval set)
scripts/selfeval.py   the measurement loop behind every CHANGELOG entry
```
