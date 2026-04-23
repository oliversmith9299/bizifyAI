# routes/main.py
# ─────────────────────────────────────────────────────────────────────────────
# AI Pipeline routes — add these to your existing routes/main.py
# Keep all your existing routes. Add this router registration in your app entry point:
#
#   from routes.main import router as pipeline_router
#   app.include_router(pipeline_router)
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db.connection import get_db
from db import crud
from agents.PipelineRunner import run_full_pipeline, chat_with_idea_agent, get_user_context

router = APIRouter(prefix="/pipeline", tags=["AI Pipeline"])

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "change-this-secret")


# ── Auth ──────────────────────────────────────────────────────────────────────
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Request Models ────────────────────────────────────────────────────────────
class QuestionnaireInput(BaseModel):
    user_id: str
    user_profile: Dict[str, Any]
    career_profile: Dict[str, Any]
    skills: List[str] = []


class ChatInput(BaseModel):
    user_id: str
    message: str


def build_questionnaire_payload(data: QuestionnaireInput) -> Dict[str, Any]:
    return {
        "user_profile": data.user_profile,
        "career_profile": data.career_profile,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    """Quick health check — backend team uses this to verify AI service is up."""
    return {"status": "ok", "timestamp": int(time.time())}


@router.post("/questionnaire", dependencies=[Depends(verify_api_key)])
def receive_questionnaire(data: QuestionnaireInput, db=Depends(get_db)):
    """
    Receive and store questionnaire output from the backend without starting the pipeline.
    Use this if the backend wants to push questionnaire data first and trigger AI later.
    """
    questionnaire = build_questionnaire_payload(data)
    crud.save_questionnaire_output(db, data.user_id, questionnaire)

    return {
        "user_id": data.user_id,
        "status": "saved",
        "message": "Questionnaire output saved.",
    }


@router.get("/questionnaire/{user_id}", dependencies=[Depends(verify_api_key)])
def get_questionnaire(user_id: str, db=Depends(get_db)):
    """Return the saved questionnaire output JSON for a user."""
    questionnaire = crud.get_questionnaire_output_json(db, user_id)
    if questionnaire is None:
        raise HTTPException(status_code=404, detail="No questionnaire found for this user_id")

    return {
        "user_id": user_id,
        "questionnaire": questionnaire,
    }


@router.post("/run", dependencies=[Depends(verify_api_key)])
async def run_pipeline(
    data: QuestionnaireInput,
    background_tasks: BackgroundTasks,
    db=Depends(get_db)
):
    """
    START the pipeline for a user.
    Call this right after the user completes the questionnaire.
    Returns 202 immediately — pipeline runs in background.
    Poll /pipeline/status/{user_id} every 3s to track progress.
    """
    questionnaire = build_questionnaire_payload(data)

    # Store the original questionnaire output JSON for later retrieval
    crud.save_questionnaire_output(db, data.user_id, questionnaire)

    # Mark as pending in DB before starting background task
    crud.upsert_pipeline_status(db, data.user_id, "pending")

    # Run pipeline in background — does NOT block the HTTP response
    background_tasks.add_task(run_full_pipeline, data.user_id, questionnaire, data.skills)

    return JSONResponse(status_code=202, content={
        "user_id": data.user_id,
        "status": "pending",
        "message": "Pipeline started. Poll /pipeline/status/{user_id} for progress.",
        "poll_url": f"/pipeline/status/{data.user_id}",
    })


@router.get("/status/{user_id}", dependencies=[Depends(verify_api_key)])
def get_status(user_id: str, db=Depends(get_db)):
    """
    Poll this after /pipeline/run.
    When idea_ready=true, call /pipeline/idea/{user_id} to show the idea.
    """
    run = crud.get_pipeline_status(db, user_id)
    if not run:
        raise HTTPException(status_code=404, detail="No pipeline found for this user_id")

    return {
        "user_id": user_id,
        "status": run.status,                                    # pending|running|done|error
        "current_step": run.current_step,                        # which step is running now
        "profile_ready":  crud.get_profile(db, user_id) is not None,
        "problems_ready": crud.get_problems(db, user_id) is not None,
        "idea_ready":     crud.get_idea(db, user_id) is not None,
        "error": run.error,
    }


@router.get("/idea/{user_id}", dependencies=[Depends(verify_api_key)])
def get_idea(user_id: str, db=Depends(get_db)):
    """
    Get the generated idea and chat history for a user.
    Only available after status=done and idea_ready=true.
    """
    idea = crud.get_idea(db, user_id)
    if not idea:
        raise HTTPException(status_code=425, detail="Idea not ready yet — pipeline still running")

    return {
        "user_id": user_id,
        "current_idea": idea.current_idea,
        "chat_history": idea.chat_history or [],
    }


@router.post("/chat", dependencies=[Depends(verify_api_key)])
def chat(data: ChatInput, db=Depends(get_db)):
    """
    Send a chat message to the idea agent.
    Call this for every message the user types in the chat UI.
    Synchronous — returns the reply immediately.
    """
    # Verify pipeline is complete before allowing chat
    run = crud.get_pipeline_status(db, data.user_id)
    if not run:
        raise HTTPException(status_code=404, detail="No pipeline found. Call /pipeline/run first.")
    if run.status != "done":
        raise HTTPException(status_code=425, detail=f"Pipeline not ready yet (status: {run.status})")

    # Get current idea from DB
    idea_row = crud.get_idea(db, data.user_id)
    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not generated yet")

    # Get conversation context (stored in memory during this server session)
    context = get_user_context(data.user_id)

    # If server restarted and context was lost, rebuild from DB
    if not context:
        profile_row  = crud.get_profile(db, data.user_id)
        problems_row = crud.get_problems(db, data.user_id)
        if profile_row and problems_row:
            from agents.PipelineRunner import _build_context
            # Minimal questionnaire reconstruction for context rebuild
            context = _build_context(
                profile_row.data,
                problems_row.data,
                {"user_profile": {}},   # region/setup already baked into constraints
                []
            )

    # Get AI reply
    reply = chat_with_idea_agent(data.user_id, data.message, context)

    # Update idea in DB if a new idea was generated
    new_idea = reply if "💡 IDEA:" in reply else idea_row.current_idea

    # Append to chat history
    existing_history = idea_row.chat_history or []
    existing_history.append({"role": "user", "content": data.message})
    existing_history.append({"role": "assistant", "content": reply})

    crud.save_idea(db, data.user_id, new_idea, existing_history)

    return {
        "user_id": data.user_id,
        "reply": reply,
        "chat_history_length": len(existing_history),
    }


@router.get("/problems/{user_id}", dependencies=[Depends(verify_api_key)])
def get_problems(user_id: str, db=Depends(get_db)):
    """Returns validated problems for a user (optional — for frontend display)."""
    row = crud.get_problems(db, user_id)
    if not row:
        raise HTTPException(status_code=425, detail="Problems not ready yet")
    return {"user_id": user_id, "problems": row.data}


@router.get("/profile/{user_id}", dependencies=[Depends(verify_api_key)])
def get_profile(user_id: str, db=Depends(get_db)):
    """Returns founder profile analysis (optional — for frontend display)."""
    row = crud.get_profile(db, user_id)
    if not row:
        raise HTTPException(status_code=425, detail="Profile not ready yet")
    return {"user_id": user_id, "profile": row.data}
