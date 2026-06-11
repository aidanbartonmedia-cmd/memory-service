"""Request/response models. Lenient on input (resilience), exact on output (contract)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _coerce_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(v)


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = "user"
    name: str | None = None
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def _content_to_str(cls, v: Any) -> str:
        return _coerce_text(v)

    @field_validator("role", mode="before")
    @classmethod
    def _role_to_str(cls, v: Any) -> str:
        return str(v) if v is not None else "user"


class TurnIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str = Field(min_length=1, max_length=512)
    user_id: str | None = Field(default=None, max_length=512)
    messages: list[MessageIn] = Field(min_length=1, max_length=200)
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None

    def parsed_timestamp(self) -> datetime:
        if self.timestamp:
            try:
                return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)


class RecallIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str = ""
    session_id: str | None = None
    user_id: str | None = None
    max_tokens: int = Field(default=1024, ge=16, le=32768)

    @field_validator("query", mode="before")
    @classmethod
    def _query_to_str(cls, v: Any) -> str:
        return _coerce_text(v)


class SearchIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str = ""
    session_id: str | None = None
    user_id: str | None = None
    limit: int = Field(default=10, ge=1, le=100)

    @field_validator("query", mode="before")
    @classmethod
    def _query_to_str(cls, v: Any) -> str:
        return _coerce_text(v)


# ---- responses (contract shapes) ----

class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallOut(BaseModel):
    context: str
    citations: list[Citation]


class SearchResult(BaseModel):
    content: str
    score: float
    session_id: str | None
    timestamp: str | None
    metadata: dict[str, Any]


class SearchOut(BaseModel):
    results: list[SearchResult]


class MemoryOut(BaseModel):
    id: str
    type: str
    key: str
    value: str
    confidence: float
    entities: list[str]
    source_session: str | None
    source_turn: str | None
    created_at: str
    updated_at: str
    supersedes: str | None
    superseded_by: str | None
    active: bool


class MemoriesOut(BaseModel):
    memories: list[MemoryOut]
