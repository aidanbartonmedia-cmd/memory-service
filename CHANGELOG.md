# CHANGELOG

Every entry below was measured with the same loop: `scripts/selfeval.py`
ingests the 5 scripted conversations in `fixtures/conversations.json` (14
turns, 5 users) and runs probe queries against `/recall`, checking expected
facts, supersession chains via `/users/{id}/memories`, and empty-context
behavior on noise probes. The probe set was 24 queries through v0.6 and grew
to 32 at v0.7 (cross-user noise + paraphrase stress probes). Raw result JSON
for every run quoted here is committed under `selfeval-results/` — scores
before v0.7 are against the 24-probe set, from v0.7 against the 32-probe set.
(The three earliest v0.1-labeled result files scoring 1/24 are harness
bootstrap runs from before the skeleton served data correctly; the quoted
v0.1 baseline is the last of them.)

A caveat I kept in mind throughout: the fixture metric is substring-based, so
it is *more lenient* than an LLM judge — a raw-chat-text context can pass a
probe that a judge would score poorly. That is exactly what v0.1's score
shows, and why I didn't stop there.

---

## v0.1 — Contract skeleton: SQLite + FTS5 keyword recall

**What changed:** All seven contract endpoints on FastAPI + SQLite (WAL,
`synchronous=FULL`). Recall = BM25 over raw turn text, rendered as dated
snippet lines. No extraction, no embeddings.

**Why:** Get the contract shape and the measurement loop working before any
intelligence. The eval harness (`scripts/selfeval.py`) was built *before*
this version — every later decision is justified by its numbers.

**Result:** Self-eval **17/24 (0.71)**, recall p50 1ms. The passing 17 are
substring luck — expected words appear in raw chat snippets. The failures are
the structural ones: noise resistance 0/3 (any keyword overlap returns junk),
fact-evolution history checks 0/2 (`/users/{id}/memories` returns nothing —
the exact "message log, not memory service" red flag), and the supersession
probes can't work because nothing is extracted.

**Next:** Extraction. The memories endpoint must show typed, keyed,
provenance-carrying facts, and contradictions must supersede.

---

## v0.2 — LLM extraction with in-context supersession decisions

**What changed:** `/turns` now runs a Claude extraction pass (structured
output, pydantic-validated). The extractor sees the user's *existing active
memories* (id, type, key, value) alongside the new turn and returns explicit
actions: `create`, `supersede` (with the target memory id), or `reinforce`.
Supersession writes a chain (`supersedes` / `superseded_by`, old row kept
inactive). Each memory carries type/key/value/confidence/entities/provenance
and a local embedding. Turns get a one-sentence third-person summary (a
retrieval target) + embedding. Rule-based fallback extractor when no API key.
Context assembly became tiered: known-facts → events → conversation snippets.

**Why:** Contradiction detection is a semantic problem, not a string problem.
"I just started at Notion" only contradicts `employment.employer = Stripe` if
something can see both at once and reason about it — so the supersession
decision belongs *inside* the extraction call, not in a post-hoc key-equality
check. (The fallback extractor does use key equality — that's why it's the
degraded path.)

**Result:** **21/24 (0.88)**, up from 17. Fact evolution 4/4 (current fact
returned, Stripe→Notion chain inspectable), corrections 1/1 (peanut→shellfish
correction never materializes a peanut-allergy memory), opinion arc 1/1,
implicit facts 2/2 ("walking Biscuit" → `pets.dog.name`). Ingest cost: 60.4s
for 14 turns (~4.3s/turn with claude-opus-4-8) — well inside the eval's 60s
per-call budget, and it buys the extraction quality this design bets on.
Noise still 0/3: the profile tier is returned for *any* query.

**Next:** Retrieval is still keyword-only; paraphrase queries score badly at
the retrieval layer (masked on the fixture by the profile tier). Then noise.

---

## v0.3 — Hybrid retrieval: local dense embeddings + BM25, RRF fusion

**What changed:** Added dense retrieval over memory and turn-summary
embeddings (bge-small-en-v1.5, 384-d, ONNX via fastembed — runs locally, no
API, model baked into the Docker image) and fused both rankings with
reciprocal rank fusion (k=60). Query-side embeddings use BGE's asymmetric
query prefix.

**Why:** Pure BM25 misses paraphrases ("where is the user based?" vs "Lives
in Berlin"); pure dense misses keyword-anchored queries ("what's their dog's
*name*?"). RRF fuses rank positions, so no score calibration between the two
systems is needed.

**Result:** Fixture score flat at **21/24** — expected, since the profile
tier already carried the paraphrase probes — but the retrieval layer itself
got measurably better: through `/search` (no profile tier), "what part of the
world is this person located in these days" (zero token overlap) now ranks
`home.city: Lives in Berlin` first. Recall p50 2ms → 5ms — acceptable.
Max-dense-similarity diagnostics plumbed out of `retrieve()` for the next
step.

**Next:** Noise resistance (0/3 since v0.1) and true multi-hop ranking.

---

## v0.4 — Entity-hop expansion + calibrated noise gate (two sub-iterations)

**What changed (1):** One-hop graph expansion: extraction tags every memory
with entities (`["biscuit","dog","pet"]`); at recall, memories sharing an
entity with a top-fused hit are pulled in at the parent's score × 0.5.
"What city does the user with the dog named Biscuit live in?" → Biscuit
memory hits directly, location memory arrives via the shared-entity hop.

**What changed (2):** Relevance gate for noise resistance. Calibrated on the
fixture: hardest noise probe peaked at dense 0.585 while the weakest real
probe sat at 0.592 — a 0.007 margin, too thin for one threshold. Two-tier
rule instead: relevant if max dense ≥ 0.62 alone, or ≥ 0.50 *with* a keyword
hit confirming. Below the gate, `/recall` returns `{"context": "",
"citations": []}`.

**What I observed (and fixed) along the way:**
- *First attempt leaked (23/24, noise 2/3):* "Tell me about the user's
  wedding plans" passed the gate — FTS5 indexes stopwords, so the query's
  "the/about/me" matched every document and "keyword confirmation" was
  vacuously true. Fix: query-side stopword stripping (including "user", which
  our own "User ..." summaries inject into every document). Porter stemming
  enabled on both FTS tables at the same time so the confirm signal can match
  "live"→"Lives".
- *That fix regressed p06 (23/24 again, different probe):* "Where does the
  user live?" returned empty. Root cause was upstream in extraction: that
  ingest had phrased the memory as "**Relocating** to Denver" (the
  transition) rather than "**Lives in** Denver" (the state) — weak dense
  match, no keyword match. Added an extraction-prompt rule: fact values are
  phrased as the current state of the world; the transition becomes a
  separate event memory. Retrieval failures can be extraction bugs.

**Result:** **24/24 (1.00)**, including noise 3/3 — and stable at 24/24 on a
further independent fresh ingest (`v0.4-stability`; extraction has
run-to-run variance, one clean run proves little — later fresh ingests at
v0.5-docker/v0.6-docker re-confirmed on the same probe set). Recall p50 5ms.

**Next:** The gate margins are honest but thin (calibrated on 24 probes, not
2,400). The floors are env-tunable (`MEMORY_DENSE_FLOOR*`) and the
calibration script is in the repo; with more fixture data I'd re-fit them
first thing.

---

## v0.5 — Budget-safe assembly, citation hygiene, global /search

**What changed:** Section-wise context building — a tier header is only
emitted if at least one of its lines fits, so tiny budgets can't produce
orphan headers; "previously: ..." history decorations are dropped below
256-token budgets (history is the first nicety to cut when every token
competes with a current fact); citations dedupe by turn (keep max score, cap
12); `/search` with null user *and* session scopes globally — an explicit
tool call is a deliberate "look everywhere".

**Why:** The budget sweep showed degenerate outputs at the edges (a header
with no content at `max_tokens=16`; duplicate citations when several memories
share a source turn).

**Result:** Still **24/24**. Budget sweep at max_tokens ∈ {16, 32, 64, 128,
256, 512, 2048}: usage ratio ≤ 0.97, never over budget (the contract allows
2×; we stay under 1×), and degradation is by priority — at 32 tokens you get
exactly the top-ranked known facts and nothing else.

**Next:** Docker verification on a clean machine path (compose up → smoke
test → compose down/up persistence), README.

---

## v0.6 — Adversarial-review hardening

**What changed:** After v0.5 I ran an adversarial review pass over the
submission (multiple independent reviewers checking contract compliance,
code correctness, eval-harness behavior, docs accuracy, and originality
exposure) and fixed what survived verification:

1. **Session→user owner resolution on reads.** A `/recall` carrying
   `user_id: null` but a `session_id` whose turns were written under a user
   previously resolved to the (empty) anonymous scope. Now it resolves to
   that user — the caller that knows the session was the one writing to it.
   Documented as a deliberate bearer-capability decision in README §4;
   anonymous-session isolation is unchanged (regression-tested).
2. **Atomic `/turns`.** Extraction + embedding now run lock-free, then the
   turn, its memories, and supersession flips apply in a *single*
   transaction. Restart-mid-write can no longer observe a turn without its
   memories or a half-flipped chain, and `/recall` is never blocked behind a
   multi-second LLM call holding the connection lock. Dead single-step write
   paths removed.
3. **Docs drift caught by review:** the README architecture diagram and
   failure-mode table still described the pre-atomic write order — rewritten
   to match the code. Added: the gate floors are calibrated for
   bge-small-en-v1.5 specifically (swapping `MEMORY_EMBEDDING_MODEL` requires
   re-calibration — method documented), and a prior-art section locating the
   design against mem0/Zep/Hindsight per the originality rule.

**Result:** pytest 30/30; self-eval re-run after the changes: **24/24** with
the same latency profile. No retrieval-quality change expected or observed —
this round was correctness, scoping, and review-readiness.

**Next:** below.

---

## v0.7 — Stress probes break the gate; term-support gate fixes it (plus two crash/deadline fixes)

**What changed:** The adversarial review's strongest finding was that the
noise gate didn't generalize: its own showcase probe ("tell me about the
user's wedding plans"), pointed at a *different* fixture user, returned the
full profile. I extended the fixture to 32 probes — the same noise queries
against every user, "favorite movie" against the favorite-cuisine user,
plus three paraphrase probes with zero token overlap ("what does this
person do to pay the bills?" → photographer) — and iterated against it:

- *Per-item conjunction* (the v0.4 gate took max-dense-anywhere + any-
  keyword-anywhere; requiring both signals on the same item fixed the
  split-evidence leak): 27/32. Calibration on the expanded set then showed
  the real problem — noise tops at 0.594 holistic similarity, genuine
  paraphrases bottom at 0.57. **No floor can separate them.**
- *Term-level support*: measured per-term — the query's topical noun,
  embedded alone against the best ambiguous-zone item, separates cleanly:
  "wedding"/"movie"/"soccer" top out at 0.496 support, "dinner"/"pay"/
  "salary"/"live" bottom at 0.55+. The frame words ("favorite", "plans")
  that caused the holistic overlap are excluded from evidence terms. First
  cut scored 28/32 but *re-leaked* three noise probes — the tokenizer kept
  "user's" as a content term (possessive regex) and it semantically matched
  everything; normalizing possessives fixed it. **32/32**, stable on a
  second independent fresh ingest. Calibration + budget artifacts committed
  (`selfeval-results/calibration-v0.7.txt`, `budget-sweep-v0.7.txt`).

**Also fixed from review findings, each with a regression test:**
- The heuristic fallback extractor crashed (IndexError) on message content
  containing colon-less lines starting with "user" — and since v0.6 ordering
  that 500'd the request and dropped the turn. It now parses the structured
  messages, and `extract()` degrades llm → heuristic → minimal-summary, never
  raising.
- Worst-case `/turns` latency could hit ~137s (45s SDK timeout × 3 attempts
  + retry-after sleeps) against the eval's 60s budget. Extraction now runs
  under a hard 40s wall-clock deadline (20s/attempt, ≤2 attempts) before
  falling back.
- Out-of-range `max_tokens`/`limit` clamp instead of 400; tiny budgets emit
  the top fact bare instead of an empty context (header didn't fit); CJK
  chars are charged a full token (the chars/4 heuristic undercounted them
  ~4×); timestamps normalize to UTC before lexicographic ordering; deletes
  null dangling `supersedes` pointers; reinforce ops are guarded (target
  must exist+be active, confidence only ratchets up); events tier only
  renders query-relevant events; `HEAD /health` supported; the spec-required
  recall-quality fixture test now lives in `tests/` (heuristic-mode floor;
  the LLM-mode loop remains `scripts/selfeval.py`).

**Result:** Self-eval **32/32 (1.00)** on the expanded stress set, two
independent fresh ingests. pytest 38/38. Budget sweep worst ratio 0.95
(budgets 8–2048). Recall p50 5ms / p95 12ms (term-support adds a few ms in
the ambiguous zone only).

**Next:** below.

---

## Where I'd go next (not built)

- **Re-fit the gate floors on a bigger probe set** — 24 probes is enough to
  catch design errors (it caught two), not enough to trust 0.62/0.50 as
  universal constants.
- **Session-scoped working memory.** Recall currently treats "recent
  conversations" as user-global. A same-session recency channel would help
  long single-session evals.
- **Embedding-based supersession backstop.** If the extractor misses a
  contradiction (it sees existing memories, but context windows aren't
  infallible), a write-time cosine check between the new memory and existing
  active ones under the same key prefix could flag missed supersessions.
- **Opinion-arc rendering.** Chains are stored and `/recall` shows
  "previously: ...", but a 3+ step arc could be summarized as an arc
  ("warmed → frustrated → pragmatic") rather than one hop of history.
