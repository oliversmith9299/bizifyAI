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
    from agents.generalBot import (
        _classify_intent,
        _load_pipeline_snapshot,
        _load_section_data,
        _respond_run_section,
        _respond_refine_section,
        _snapshot_to_context_string,
        OUT_OF_SCOPE_RESPONSE,
    )
    from db.connection import SessionLocal as _SL
    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.general_bot_prompt import GENERAL_BOT_SYSTEM_PROMPT

    with _SL() as db:
        snapshot = _load_pipeline_snapshot(data.user_id, db)

    classification = _classify_intent(data.message, data.history, snapshot)
    intent  = classification.get("intent", "general_startup_chat")
    section = classification.get("section")

    if intent in ("run_section", "refine_section", "start_pipeline"):
        if intent == "run_section":
            reply_text, route = _respond_run_section(section, snapshot)
        elif intent == "refine_section":
            reply_text, route = _respond_refine_section(section, snapshot)
        else:
            is_fresh   = not any(v["done"] for v in snapshot["sections"].values())
            reply_text = (
                "To start a new pipeline, your backend needs to call "
                "POST /pipeline/run with your questionnaire data. "
                + (
                    "You don't have any sections generated yet — this is the right first step."
                    if is_fresh else
                    f"You already have {snapshot['completed_count']} sections. "
                    "Starting over will replace them."
                )
            )
            route = "/pipeline/run"

        trigger = route.replace("{user_id}", data.user_id) if route else None

        def _instant():
            yield f"data: {json.dumps({'type': 'token', 'content': reply_text})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'intent': intent, 'action': intent, 'route_to_trigger': trigger, 'trigger_needed': trigger is not None})}\n\n"

        return StreamingResponse(_instant(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if intent == "out_of_scope":
        def _decline():
            yield f"data: {json.dumps({'type': 'token', 'content': OUT_OF_SCOPE_RESPONSE})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'intent': intent, 'action': 'declined', 'route_to_trigger': None, 'trigger_needed': False})}\n\n"

        return StreamingResponse(_decline(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    with _SL() as db:
        snapshot     = _load_pipeline_snapshot(data.user_id, db)
        section_data = _load_section_data(data.user_id, section, db) if section else None

    context_parts = [_snapshot_to_context_string(snapshot)]
    if section_data and section:
        context_parts.append(
            f"\n=== DETAILED DATA FOR [{section}] ===\n"
            + json.dumps(section_data, indent=2)[:4000]
        )

    messages = [
        {"role": "system", "content": GENERAL_BOT_SYSTEM_PROMPT},
        {"role": "system", "content": "\n\n".join(context_parts)},
        *data.history[-20:],
        {"role": "user",   "content": data.message},
    ]

    def _on_complete(full_reply: str) -> Dict[str, Any]:
        return {"intent": intent, "action": "answered", "route_to_trigger": None, "trigger_needed": False}

    return sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=600, on_complete=_on_complete,
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
