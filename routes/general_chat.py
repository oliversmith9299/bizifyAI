"""
routes/general_chat.py
=======================
General-purpose startup assistant chat endpoints.

/general-chat        — synchronous (waits for full response, including agent runs)
/general-chat/stream — SSE streaming; for "run_section" intents a "thinking"
                       token is emitted immediately so the client gets instant
                       feedback while the agent executes in the background.
/explain             — section-specific explainer bot
"""

import json
from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from routes.dependencies import verify_api_key, sse_chat_stream, GeneralBotInput, ExplainerInput

router = APIRouter()


@router.post("/general-chat", dependencies=[Depends(verify_api_key)])
def general_chat(data: GeneralBotInput) -> Dict[str, Any]:
    from agents.generalBot import run_general_bot

    result = run_general_bot(
        user_id=data.user_id,
        message=data.message,
        history=data.history,
    )
    return {"user_id": data.user_id, **result, "chat_history_length": len(data.history) + 2}


@router.post("/general-chat/stream", dependencies=[Depends(verify_api_key)])
def general_chat_stream(data: GeneralBotInput) -> StreamingResponse:
    """
    SSE streaming variant.

    For intents that run agents (run_section / confirm_action) the response
    cannot be token-streamed because the agent itself is doing the LLM work
    internally.  Instead we:
      1. Emit a brief "thinking" token immediately so the UI shows activity.
      2. Run the bot (which may take 30–60 s if agents are executing).
      3. Emit the full reply as a final token followed by the done event.

    For all other intents (chat_about_data, general_startup_chat, etc.) we
    do true token-by-token streaming via sse_chat_stream.
    """
    from agents.generalBot import (
        _classify_intent,
        _load_pipeline_snapshot,
        _load_section_data,
        _snapshot_to_context,
        run_general_bot,
    )
    from agents.config import client as groq_client, GROQ_MODEL
    from System_Messages.general_bot_prompt import GENERAL_BOT_SYSTEM_PROMPT
    from db.connection import SessionLocal as _SL

    # ── Quick pre-classification to decide streaming strategy ─────────────────
    with _SL() as db:
        snapshot = _load_pipeline_snapshot(data.user_id, db)

    classification = _classify_intent(data.message, data.history, snapshot)
    intent = classification.get("intent", "general_startup_chat")
    section = classification.get("section")

    # Intents that run agents block for a long time → use the "thinking" pattern
    _agent_running_intents = {"run_section", "confirm_action"}

    if intent in _agent_running_intents:
        def _agent_stream():
            # Immediate feedback while the agent is working
            thinking = "Give me a moment while I work on that for you..."
            yield f"data: {json.dumps({'type': 'token', 'content': thinking})}\n\n"

            # Run the bot (blocking — agent executes here)
            try:
                result = run_general_bot(
                    user_id=data.user_id,
                    message=data.message,
                    history=data.history,
                )
                reply = result["reply"]
                # Replace the thinking message with the actual reply
                yield f"data: {json.dumps({'type': 'replace', 'content': reply})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'intent': result['intent'], 'action': result['action'], 'route_to_trigger': None, 'trigger_needed': False})}\n\n"
            except Exception as e:
                error_msg = "Something went wrong while running the analysis. Please try again."
                yield f"data: {json.dumps({'type': 'replace', 'content': error_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'intent': intent, 'action': 'error', 'route_to_trigger': None, 'trigger_needed': False})}\n\n"

        return StreamingResponse(
            _agent_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── All other intents: true token streaming ───────────────────────────────
    with _SL() as db:
        snapshot     = _load_pipeline_snapshot(data.user_id, db)
        section_data = _load_section_data(data.user_id, section, db) if section else None

    context_parts = [_snapshot_to_context(snapshot)]
    if section_data and section:
        context_parts.append(
            f"\n=== {section} DATA ===\n"
            + json.dumps(section_data, indent=2)[:4000]
        )

    messages = [
        {"role": "system", "content": GENERAL_BOT_SYSTEM_PROMPT},
        {"role": "system", "content": "\n\n".join(context_parts)},
        *data.history[-20:],
        {"role": "user",   "content": data.message},
    ]

    def _on_complete(full_reply: str) -> Dict[str, Any]:
        return {
            "intent":           intent,
            "action":           "answered",
            "route_to_trigger": None,
            "trigger_needed":   False,
        }

    return sse_chat_stream(
        messages=messages,
        groq_client=groq_client,
        groq_model=GROQ_MODEL,
        temperature=0.4,
        max_tokens=600,
        on_complete=_on_complete,
    )


@router.post("/explain", dependencies=[Depends(verify_api_key)])
def explain(data: ExplainerInput) -> Dict[str, Any]:
    from agents.ExplainerBot import run_explainer_bot

    result = run_explainer_bot(
        user_id=data.user_id,
        message=data.message,
        history=data.history,
        section=data.section,
    )
    return {"user_id": data.user_id, **result, "chat_history_length": len(data.history) + 2}