"""HTTP layer: the seven contract endpoints, auth, and resilience guards."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import assembly, config, db, embeddings, extraction, retrieval, store
from .models import (
    MemoriesOut,
    MemoryOut,
    RecallIn,
    RecallOut,
    SearchIn,
    SearchOut,
    SearchResult,
    TurnIn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("memory.app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    yield
    db.close()


app = FastAPI(title="memory-service", lifespan=lifespan)


# ---------- resilience: never crash, always JSON ----------

@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "invalid_request", "detail": exc.errors()[:5]})


@app.exception_handler(StarletteHTTPException)
async def http_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "internal_error"})


@app.middleware("http")
async def guards(request: Request, call_next):
    # Oversized payloads -> 413, before body parsing.
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > config.MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"error": "payload_too_large"})
    start = time.monotonic()
    response = await call_next(request)
    log.info("%s %s -> %s (%.0fms)", request.method, request.url.path,
             response.status_code, (time.monotonic() - start) * 1000)
    return response


# ---------- optional bearer auth ----------

async def auth(request: Request) -> None:
    if config.AUTH_TOKEN is None:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {config.AUTH_TOKEN}":
        raise StarletteHTTPException(status_code=401, detail="unauthorized")


# ---------- endpoints ----------

@app.get("/health")
async def health() -> JSONResponse:
    if not db.ready():
        return JSONResponse(status_code=503, content={"status": "starting"})
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.post("/turns", status_code=201, dependencies=[Depends(auth)])
def post_turn(turn: TurnIn) -> dict[str, str]:
    ts = turn.parsed_timestamp().isoformat()
    messages = [m.model_dump() for m in turn.messages]
    # 1. Persist the raw turn first — extraction failures must never lose data.
    turn_id = store.add_turn(
        session_id=turn.session_id,
        user_id=turn.user_id,
        ts=ts,
        messages=messages,
        metadata=turn.metadata,
    )
    owner = store.owner_key(turn.user_id, turn.session_id)

    # 2. Extract structured memories (synchronously — the contract requires
    #    read-after-write visibility when this endpoint returns).
    existing = store.get_memories(owner, active_only=True)
    result, mode = extraction.extract(messages, existing, ts)

    applied = 0
    for em in result.memories:
        try:
            if em.action == "reinforce" and em.supersedes_id:
                store.touch_memory(em.supersedes_id, confidence=em.confidence)
                continue
            supersedes_id = em.supersedes_id if em.supersedes_id else None
            vec = embeddings.embed_one(f"{em.key}: {em.value}")
            store.insert_memory(
                owner=owner,
                user_id=turn.user_id,
                type_=em.type,
                key=em.key,
                value=em.value,
                confidence=em.confidence,
                entities=em.entities,
                source_session=turn.session_id,
                source_turn=turn_id,
                supersedes_id=supersedes_id,
                embedding=embeddings.to_blob(vec),
            )
            applied += 1
        except Exception:
            log.exception("failed to apply extracted memory %s", em.key)

    # 3. Enrich the turn with its summary (a retrieval target) + embedding.
    summary = result.turn_summary.strip()
    turn_vec = embeddings.embed_one(summary) if summary else None
    store.enrich_turn(turn_id, summary=summary, embedding=embeddings.to_blob(turn_vec))

    log.info("turn %s stored (session=%s user=%s) extraction=%s memories=%d",
             turn_id, turn.session_id, turn.user_id, mode, applied)
    return {"id": turn_id}


@app.post("/recall", dependencies=[Depends(auth)])
def recall(req: RecallIn) -> RecallOut:
    owner = store.owner_key(req.user_id, req.session_id)
    retrieved = retrieval.retrieve(owner, req.query)
    # Noise resistance: when nothing in the store is plausibly about this
    # query, return an empty context rather than the user's profile — a
    # frozen LLM treats whatever we inject as relevant, so irrelevant facts
    # invite hallucinated connections. (/search stays ungated: an explicit
    # tool call gets best-effort ranked results.)
    if not retrieved["relevant"]:
        log.info("recall gated to empty (owner=%s, diag=%s)",
                 owner, {k: v for k, v in retrieved["diagnostics"].items()
                         if not isinstance(v, dict)})
        return RecallOut(context="", citations=[])
    context, citations = assembly.assemble(
        owner=owner, query=req.query, retrieved=retrieved, max_tokens=req.max_tokens
    )
    return RecallOut(context=context, citations=citations)


@app.post("/search", dependencies=[Depends(auth)])
def search(req: SearchIn) -> SearchOut:
    owner = store.owner_key(req.user_id, req.session_id)
    retrieved = retrieval.retrieve(owner, req.query, limit=req.limit)
    results: list[SearchResult] = []
    for mem_id, score in retrieved["memories"][: req.limit]:
        mem = store.get_memory(mem_id)
        if not mem:
            continue
        results.append(SearchResult(
            content=f"{mem['key']}: {mem['value']}",
            score=round(score, 4),
            session_id=mem.get("source_session"),
            timestamp=mem.get("updated_at"),
            metadata={"kind": "memory", "type": mem["type"], "id": mem["id"],
                      "confidence": mem["confidence"], "active": bool(mem["active"])},
        ))
    for turn_id, score in retrieved["turns"][: req.limit]:
        turn = store.get_turn(turn_id)
        if not turn:
            continue
        results.append(SearchResult(
            content=assembly.turn_snippet(turn, max_chars=400),
            score=round(score, 4),
            session_id=turn.get("session_id"),
            timestamp=turn.get("ts"),
            metadata={"kind": "turn", "id": turn["id"]},
        ))
    results.sort(key=lambda r: r.score, reverse=True)
    return SearchOut(results=results[: req.limit])


@app.get("/users/{user_id}/memories", dependencies=[Depends(auth)])
def user_memories(user_id: str) -> MemoriesOut:
    rows = store.get_memories(user_id)
    out = []
    for r in rows:
        try:
            entities = json.loads(r.get("entities_json") or "[]")
        except json.JSONDecodeError:
            entities = []
        out.append(MemoryOut(
            id=r["id"], type=r["type"], key=r["key"], value=r["value"],
            confidence=r["confidence"], entities=entities,
            source_session=r.get("source_session"), source_turn=r.get("source_turn"),
            created_at=r["created_at"], updated_at=r["updated_at"],
            supersedes=r.get("supersedes"), superseded_by=r.get("superseded_by"),
            active=bool(r["active"]),
        ))
    return MemoriesOut(memories=out)


@app.delete("/sessions/{session_id}", status_code=204, dependencies=[Depends(auth)])
def delete_session(session_id: str) -> Response:
    store.delete_session(session_id)
    return Response(status_code=204)


@app.delete("/users/{user_id}", status_code=204, dependencies=[Depends(auth)])
def delete_user(user_id: str) -> Response:
    store.delete_user(user_id)
    return Response(status_code=204)
