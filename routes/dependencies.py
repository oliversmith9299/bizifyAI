"""
routes/dependencies.py
=======================
Shared FastAPI dependencies, Pydantic request models, and the SSE streaming
helper used by every route file. Import from here instead of duplicating.
"""

import json
from collections.abc import Generator
from typing import Any, Callable, Dict, List, Optional

from fastapi import Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────────────────────
# SECURITY CONTRACT:
# This service must NEVER be publicly accessible.
# All requests must come from the backend server, which:
#   1. Validates the user's JWT token
#   2. Replaces user_id in the request body with the real authenticated user ID
#   3. Adds the X-API-KEY header before forwarding here
#
# The API key proves the request came from the backend.
# It does NOT prove user_id is valid — that is the backend's job.

from agents.config import API_SECRET_KEY


def verify_api_key(x_api_key: str = Header(...)) -> None:
    """FastAPI dependency — rejects requests missing or with wrong API key."""
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Request models ────────────────────────────────────────────────────────────
# All user-supplied strings are bounded to prevent DoS via oversized payloads.
# history is capped at 100 turns (~50 exchanges) — enough for any real session.

_USER_ID   = Field(..., min_length=1,  max_length=64)
_MESSAGE   = Field(..., min_length=1,  max_length=10_000)
_PROMPT    = Field(..., min_length=1,  max_length=5_000)
_SECTION   = Field(None,               max_length=64)
_HISTORY   = Field(default_factory=list, max_length=100)


class QuestionnaireInput(BaseModel):
    user_id: str = _USER_ID


class ChatInput(BaseModel):
    user_id: str = _USER_ID
    message: str = _MESSAGE


class SectionChatInput(BaseModel):
    user_id: str            = _USER_ID
    message: str            = _MESSAGE
    history: List[Dict[str, Any]] = _HISTORY


class RegenerateCustomInput(BaseModel):
    user_id:       str = _USER_ID
    custom_prompt: str = _PROMPT


class IdeaIntakeInput(BaseModel):
    user_id: str            = _USER_ID
    message: str            = _MESSAGE
    history: List[Dict[str, Any]] = _HISTORY


class UserIdInput(BaseModel):
    user_id: str = _USER_ID


class GeneralBotInput(BaseModel):
    user_id: str            = _USER_ID
    message: str            = _MESSAGE
    history: List[Dict[str, Any]] = _HISTORY


class ExplainerInput(BaseModel):
    user_id: str            = _USER_ID
    message: str            = _MESSAGE
    history: List[Dict[str, Any]] = _HISTORY
    section: Optional[str]  = _SECTION


# ── SSE streaming helper ──────────────────────────────────────────────────────

def sse_chat_stream(
    messages:    list,
    groq_client: Any,
    groq_model:  str,
    temperature: float,
    max_tokens:  int,
    on_complete: Callable[[str], Dict[str, Any]],
) -> StreamingResponse:
    """
    Stream LLM tokens as Server-Sent Events and call `on_complete` once done.

    SSE event format
    ────────────────
    Content token : data: {"type": "token", "content": "word "}
    Stream done   : data: {"type": "done", ...on_complete result}

    All validation must happen BEFORE calling this function so HTTP errors
    are returned normally, not buried inside the SSE stream.
    """
    def _generator() -> Generator[str, None, None]:
        stream = groq_client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        tokens: list[str] = []
        total_tokens = 0
        for chunk in stream:
            if chunk.usage:
                total_tokens = chunk.usage.total_tokens or 0
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                tokens.append(delta)
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

        full_reply = "".join(tokens)
        metadata   = on_complete(full_reply)
        yield f"data: {json.dumps({'type': 'done', 'tokens_used': total_tokens, **metadata})}\n\n"

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

