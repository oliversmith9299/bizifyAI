import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db.connection import get_db
from db import crud


router = APIRouter(prefix="/pipeline", tags=["AI Pipeline"])

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-key")

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
        "skills": data.skills  # FIX: Skills are now preserved
    }

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "timestamp": int(time.time())}

#----------------------------------------------------------------

@router.post("/run", dependencies=[Depends(verify_api_key)])
async def run_pipeline(
    data: QuestionnaireInput,
    background_tasks: BackgroundTasks,
    db=Depends(get_db)
):
    """
    START the pipeline for a user.
    Backend sends all questionnaire data -> we save it -> run pipeline in background.
    """
    questionnaire = build_questionnaire_payload(data)

    crud.save_questionnaire_output(db, data.user_id, questionnaire)

    # Mark as pending in DB
    crud.upsert_pipeline_status(db, data.user_id, "pending")

    # Import pipeline here to avoid circular dependencies if any
    from agents.PipelineRunner import run_full_pipeline
    background_tasks.add_task(run_full_pipeline, data.user_id, questionnaire, data.skills)

    return JSONResponse(status_code=202, content={
        "user_id": data.user_id,
        "status": "pending",
        "message": "Pipeline started. Poll /pipeline/status/{user_id} for progress.",
        "poll_url": f"/pipeline/status/{data.user_id}",
    })



#---------------------------------------------------------------------------------------
@router.get("/status/{user_id}", dependencies=[Depends(verify_api_key)])
def get_status(user_id: str, db=Depends(get_db)):
    """Poll this after /pipeline/run."""
    run = crud.get_pipeline_status(db, user_id)
    if not run:
        raise HTTPException(status_code=404, detail="No pipeline found for this user_id")

    return {
        "user_id": user_id,
        "status": run.status,
        "current_step": run.current_step,
        "profile_ready":  crud.get_profile(db, user_id) is not None,
        "problems_ready": crud.get_problems(db, user_id) is not None,
        "idea_ready":     crud.get_idea(db, user_id) is not None,
        "error": run.error,
    }


#--------------------------------------------------------------------------------
@router.get("/idea/{user_id}", dependencies=[Depends(verify_api_key)])
def get_idea(user_id: str, db=Depends(get_db)):
    """Get the generated idea and chat history for a user."""
    idea = crud.get_idea(db, user_id)
    if not idea:
        raise HTTPException(status_code=425, detail="Idea not ready yet — pipeline still running")

    return {
        "user_id": user_id,
        "current_idea": idea.current_idea,
        "chat_history": idea.chat_history or [],
    }

#--------------------------------------------------------------------------------
@router.post("/chat", dependencies=[Depends(verify_api_key)])
def chat(data: ChatInput, db=Depends(get_db)):
    """
    Send a chat message to the idea agent.
    DB-Driven context — No memory states.
    """
    # 1. Verify pipeline state
    run = crud.get_pipeline_status(db, data.user_id)
    if not run or run.status != "done":
        raise HTTPException(status_code=425, detail="Pipeline not ready yet. Call /pipeline/run first.")

    # 2. Pull all context directly from DB (No memory fallbacks)
    idea_row = crud.get_idea(db, data.user_id)
    profile_row = crud.get_profile(db, data.user_id)
    problems_row = crud.get_problems(db, data.user_id)
    questionnaire_row = crud.get_questionnaire_output(db, data.user_id)

    if not (idea_row and profile_row and problems_row and questionnaire_row):
        raise HTTPException(status_code=425, detail="Missing required context in DB to chat.")

    # 3. Rebuild context explicitly
    from agents.PipelineRunner import _build_context
    stored_questionnaire = questionnaire_row.data
    skills = stored_questionnaire.get("skills", [])

    context = _build_context(
        profile_row.data,
        problems_row.data,
        stored_questionnaire, # Pass FULL saved questionnaire
        skills
    )

    # 4. Generate AI Reply
    from agents.PipelineRunner import groq_client, GROQ_MODEL, IDEA_SYSTEM_PROMPT
    
    existing_history = idea_row.chat_history or []
    messages = [
        {"role": "system", "content": IDEA_SYSTEM_PROMPT},
        {"role": "system", "content": context},
        *existing_history[-20:], # Only pass recent history to avoid token limits
        {"role": "user", "content": data.message}
    ]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1000,
    )
    reply = response.choices[0].message.content.strip()

    # 5. Update DB
    new_idea = reply if "💡 IDEA:" in reply else idea_row.current_idea
    existing_history.append({"role": "user", "content": data.message})
    existing_history.append({"role": "assistant", "content": reply})

    crud.save_idea(db, data.user_id, new_idea, existing_history)

    return {
        "user_id": data.user_id,
        "reply": reply,
        "chat_history_length": len(existing_history),
    }

#────────────────────────────────────────────────────────────────────────────
@router.get("/questionnaire/{user_id}", dependencies=[Depends(verify_api_key)])
def get_questionnaire(user_id: str, db=Depends(get_db)):
    row = crud.get_questionnaire_output(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No questionnaire found")
    
    return {
        "user_id": user_id,
        "questionnaire": row.data
    }

#────────────────────────────────────────────────────────────────────────────
@router.get("/version-check")
def version_check():
    return {"version": "NEW_CODE_123"}


#────────────────────────────────────────────────────────────────────────────
@router.get("/profile/{user_id}", dependencies=[Depends(verify_api_key)])
def get_profile(user_id: str, db=Depends(get_db)):
    profile = crud.get_profile(db, user_id)

    if not profile:
        raise HTTPException(status_code=425, detail="Profile not ready yet")

    return {
        "user_id": user_id,
        "profile_analysis": profile.data
    }


#────────────────────────────────────────────────────────────────────────────

@router.get("/problems/{user_id}", dependencies=[Depends(verify_api_key)])
def get_problems(user_id: str, db=Depends(get_db)):
    problems = crud.get_problems(db, user_id)

    if not problems:
        raise HTTPException(status_code=425, detail="Problems not ready yet")

    return {
        "user_id": user_id,
        "problems": problems.data
    }
#────────────────────────────────────────────────────────────────────────────
@router.post("/rerun/profile/{user_id}", dependencies=[Depends(verify_api_key)])
def rerun_profile(user_id: str, db=Depends(get_db)):
    from agents.PipelineRunner import run_profile_analysis

    questionnaire_row = crud.get_questionnaire_output(db, user_id)

    if not questionnaire_row:
        raise HTTPException(404, "No questionnaire")

    data = questionnaire_row.data
    skills = data.get("skills", [])

    profile = run_profile_analysis(data, skills)

    crud.save_profile(db, user_id, profile)

    return {"status": "profile regenerated"}

#────────────────────────────────────────────────────────────────────────────
@router.post("/rerun/problems/{user_id}", dependencies=[Depends(verify_api_key)])
def rerun_problems(user_id: str, db=Depends(get_db)):
    from agents.PipelineRunner import run_problem_discovery

    profile = crud.get_profile(db, user_id)
    questionnaire = crud.get_questionnaire_output(db, user_id)

    if not profile or not questionnaire:
        raise HTTPException(400, "Missing data")

    problems = run_problem_discovery(profile.data, questionnaire.data)

    crud.save_problems(db, user_id, problems)

    return {"status": "problems regenerated"}