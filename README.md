# memory-service

A memory backend for an AI agent: ingests conversation turns, extracts
structured knowledge with an LLM, and answers recall queries with hybrid
retrieval under a token budget — with fact supersession, multi-hop entity
linking, and calibrated noise resistance.

```bash
cp .env.example .env   # then uncomment + set ANTHROPIC_API_KEY (recommended)
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
         │ 1. LLM extraction       │          │ 1. hybrid retrieval          │
         │    (Claude, structured  │          │    BM25 (FTS5+porter) ───┐   │
         │    output) sees existing│          │    dense (bge-small) ────┤   │
         │    ACTIVE memories →    │          │      → RRF fusion        │   │
         │    create / supersede / │          │ 2. + one entity hop      │   │
         │    reinforce            │          │    (shared-tag expansion)│   │
         ├─────────────────────────┤          │ 3. relevance gate        │   │
         │ 2. embed memories + turn│          │    (calibrated two-tier  │   │
         │    summary (local ONNX) │          │    floor → "" on noise)  │   │
         │    — no DB lock held    │          │ 4. tiered assembly under │   │
         ├─────────────────────────┤          │    token budget          │   │
         │ 3. apply turn + memories│          │    facts → events → conv │   │
         │    + chain flips in ONE │          └──────────────┬───────────┘   │
         │    transaction          │                         │               │
         └────────────┬────────────┘                         │               │
                      ▼                                      ▼               │
         ┌──────────────────────────────────────────────────────────────────┐
         │            SQLite (WAL, synchronous=FULL) on /data volume        │
         │   turns ─ memories(+supersession chain) ─ FTS5 ─ embeddings      │
         └──────────────────────────────────────────────────────────────────┘
```

One process, one container, one store. All LLM spend happens at **write
time** (`/turns` has a 60s budget and writes are rare); **read time is
LLM-free** (recall p50 ≈ 5ms on the self-eval fixture). The slow work
(extraction, embedding) runs lock-free; the fast work (all SQL) applies in a
single transaction — when `/turns` returns 201, the turn, its memories, and
any supersession flips are all visible to `/recall` and
`/users/{id}/memories`, and a kill mid-write leaves zero partial state. No
queues, no eventual consistency, by construction.

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
4. **Relevance gate** (noise resistance): an item is evidence on its own if
   its dense similarity ≥ 0.62. In the ambiguous band [0.50, 0.62) —
   where, measured on the stress fixture, genuine paraphrases ("what does
   this person do to pay the bills?") and frame-matched noise ("what's the
   user's favorite movie?" hitting *favorite cuisine*) fully overlap — the
   item must additionally show **term-level support**: one of the query's
   content terms, embedded alone, must clear 0.52 cosine against that same
   item. Frame words (favorite, plans, name, kind...) are excluded from
   evidence terms — they're why the noise item scored high holistically in
   the first place. On the fixture this separates cleanly: noise topical
   terms (wedding, movie, soccer) top out at 0.496 support, signal terms
   (dinner, pay, salary, live) bottom at 0.55+. No evidence →
   `{"context": "", "citations": []}`, 200, never an error.

   The gate went through three measured designs (single floor: 0.007
   margin; global dense + any keyword: leaked via split evidence; per-item
   dense + keyword: leaked via frame-word keywords) — the failures and
   calibration data for each are in CHANGELOG v0.4/v0.7, and the committed
   calibration artifact is `selfeval-results/calibration-v0.7.txt`,
   reproducible via `scripts/calibrate_gate.py`. All three floors are
   calibrated *for bge-small-en-v1.5's similarity distribution*: change
   `MEMORY_EMBEDDING_MODEL` and you must re-run the calibration and reset
   `MEMORY_DENSE_FLOOR*` / `MEMORY_TERM_FLOOR`. Gating applies to `/recall`
   only — `/search` is an explicit tool call and stays best-effort.
5. **Tiered assembly under the budget** (≈4 chars/token for ASCII, 1
   token/char for non-ASCII so dense scripts can't breach the bound;
   contract allows 2×, measured usage stays ≤ 0.95×):

   - **A. Known facts about this user** — all active facts/preferences/
     opinions, ordered fact→preference→opinion, then query-relevance, then
     confidence. Stable facts first because they are the highest value-per-
     token for a frozen LLM and they're small.
   - **B. Relevant memories** — dated events.
   - **C. Relevant from recent conversations** — turn summaries by fused score.

   Priority under pressure = exactly that order (the spec's order). Cuts are
   whole-line, bottom-up within a tier; a tier header is only emitted if at
   least one line fits; "previously: …" history decorations are the first
   nicety dropped (budgets < 256); below header-sized budgets the single
   top-priority fact is emitted bare. At `max_tokens=16` you get the most
   load-bearing known fact and nothing else.

   Citations map every rendered line to its source turn (`source_turn` for
   memories, the turn itself for tier C), deduped by turn keeping max score.

**Scoping rule:** memories are keyed by `user_id` and intentionally shared
across that user's sessions (that's what makes session-2 recall of a
session-1 fact work — and the eval's own smoke test expects it). Anonymous
turns (`user_id: null`) are scoped to `session:<session_id>` and can never
bleed across sessions. A recall that passes `user_id: null` but names a
session whose turns were written under a user resolves to that user's scope —
the session demonstrably belongs to them. The implication is deliberate: a
`session_id` is treated as a bearer capability for its user's scope (the
caller that knows the session was the one writing it); deployments with
untrusted callers should set `MEMORY_AUTH_TOKEN`. Concurrent users are
isolated by the same key; the test suite hammers this in parallel threads.
One more deliberate scope decision: `/search` with `user_id` *and*
`session_id` both null searches **globally across all owners** — `/search`
models an explicit agent tool call, so a scopeless search is read as a
deliberate "look everywhere" rather than an error. `/recall` never does
this; it always resolves to exactly one owner scope.

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
  state-phrasing rules, and independent full-ingest stability runs (32/32
  twice at v0.7).
- **SQLite over a vector DB**: transactional supersession + read-after-write
  beats ANN performance we don't need at this scale (§2).
- **Brute-force cosine** is O(n) per query — correct until ~10⁵ memories per
  user, which a conversational agent won't hit; the seam to swap is one module.
- **Conservative noise gate**: I tuned to prefer empty context over plausible
  junk, because a frozen LLM treats injected context as true — false context
  is worse than no context. The margins are honest-but-thin (calibrated on 24
  probes); floors are env-tunable and the calibration method is in the repo.
- **chars/4 token approximation**: dependency-free and within the contract's
  2× tolerance (measured ≤ 0.95×, with non-ASCII characters charged a full
  token each so CJK text can't silently breach the bound); exact
  tokenization would add a tokenizer dependency to save tokens nobody is
  losing.

## 7. Failure modes

| Condition | Behavior |
|---|---|
| No `ANTHROPIC_API_KEY` | Service runs; extraction degrades to the rule-based fallback (regex patterns for employment/location/pets/allergies/preferences, supersession by exact key collision). Recall, persistence, contract all intact. Logged at startup-adjacent call sites. |
| Anthropic API down / slow / over quota | Per-turn degradation under a **hard 40s wall-clock deadline** on the whole extraction call (the eval gives `/turns` 60s total; per-attempt timeouts and retry-after sleeps compose multiplicatively, so only a request-level deadline actually bounds the tail — 20s SDK timeout × ≤2 attempts, capped at 40s, then heuristic fallback). `POST /turns` still returns 201, inside the budget, with whatever the fallback extracted. |
| Embedding model unavailable (corrupt cache; it's baked into the image so this is exotic) | Dense retrieval and the dense gate disable; BM25-only retrieval with keyword-evidence gating. Weaker paraphrase recall and noise resistance, no crash. |
| Cold store / unknown user | `200 {"context": "", "citations": []}` — never an error. |
| Empty query for a known user | Returns the known-facts profile: an empty query is read as "give me the standing context", which is what an agent priming a fresh turn wants. (Deliberate — and distinct from noise queries, which return empty.) |
| Malformed JSON, missing fields, wrong types, unicode oddities, FTS metacharacters | 400 (422-class) with a JSON error body; fuzz cases in the test suite. |
| Oversized payload | 413 before body parsing (default cap 8MB). |
| Kill mid-write / `docker compose down` mid-write | A turn applies as **one** transaction (turn + memories + supersession flips), so WAL either commits all of it or rolls all of it back — never a turn without its memories or a half-flipped chain. A client that never received its 201 has, contractually, written nothing. Committed turns survive (restart tests). |
| Slow disk | Everything is synchronous, so slow disk = slower 201s, not inconsistency. `/recall` does one read transaction. |

## 8. How to run the tests

```bash
# Contract / robustness / persistence / scoping / auth / recall-quality —
# fast, no API key, no network (runs extraction in heuristic mode; the
# recall-quality test reports its X-of-Y metric and asserts a degraded-mode
# floor):
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q          # 38 tests, ~2s

# Full recall-quality self-eval (the iteration loop; uses the LLM
# extractor, needs ANTHROPIC_API_KEY in the service environment):
docker compose up -d
.venv/bin/python scripts/selfeval.py          # ingests fixtures/, runs 32 probes

# Gate calibration + budget compliance (committed artifacts in
# selfeval-results/):
PYTHONPATH=src .venv/bin/python scripts/calibrate_gate.py
.venv/bin/python scripts/budget_sweep.py
```

`scripts/selfeval.py` prints per-category pass rates and writes a JSON report
to `selfeval-results/` — the numbers quoted in `CHANGELOG.md` are those
files, committed in this repo. Current state: **32/32** (fact evolution,
multi-hop, corrections, opinion arc, implicit facts, 8 noise probes
including cross-user stress cases, paraphrase, keyword, cold-session),
stable across two independent fresh ingests at v0.7, recall p50 ≈ 5ms,
budget usage ≤ 0.95× at every budget from 8 to 2048 tokens
(`selfeval-results/budget-sweep-v0.7.txt`).

---

## Prior art & originality

I read the public designs the brief names (mem0, Zep/Graphiti, Letta,
Hindsight, Honcho) before building, and built from scratch. Where the result
converges with the field, it's worth being precise about why:

- **API shape** — dictated by the challenge contract (endpoints, the
  `fact|preference|opinion|event` enum, `supersedes`/`active`, even the
  context-format example), so any resemblance there is the spec's, not a lift.
- **Extraction: nearest analog is mem0's update phase**, which also renders
  existing memories with ids and has an LLM return add/update/delete
  decisions. The differences are load-bearing, not cosmetic: here extraction
  and reconciliation are **one fused call** (the contradiction decision sees
  the raw utterance, not a pre-extracted candidate); the reconciliation
  context is **all** active memories, not a top-k-similar retrieval that can
  miss the contradicted fact; there is deliberately **no delete** — an
  append-only doubly-linked chain with `active` flags, shaped for this
  spec's inspectability requirement; and `reinforce` is a first-class
  confidence-ratcheting action rather than a no-op.
- **Recall: nearest analog is Zep/Graphiti's "land and expand"** (BM25 +
  cosine fused with RRF, then graph expansion from initial hits) — and
  hybrid+RRF itself is textbook IR, deliberately so; the brief asks for
  something deliberate, not exotic. The differences: expansion here is over
  **flat extraction-time entity tags**, exactly one hop, with parent×0.5
  damping chosen so second-order hops fall below the relevance floor — no
  graph store, no typed/temporal edges, no BFS frontier.
- **No public analog I know of**: the **term-support relevance gate**
  (ambiguous-zone items must show per-term semantic support from the query's
  content words, frame words excluded — calibrated, with the data and all
  three failed predecessor designs committed in CHANGELOG v0.4/v0.7), and
  the `DELETE /sessions/{id}` supersession **chain repair** (deleting the
  superseding session *reactivates* the prior fact — purpose-built for this
  eval's cleanup semantics).

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
fixtures/        5 scripted conversations + 32 probes (the self-eval set)
scripts/selfeval.py        the measurement loop behind every CHANGELOG entry
scripts/calibrate_gate.py  re-fit the gate floors (run after changing models)
scripts/budget_sweep.py    verify max_tokens compliance across budgets
```
