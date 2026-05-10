import json
import os
import time
from collections.abc import Generator
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from db.connection import get_db, SessionLocal
from db import crud


router = APIRouter(prefix="/pipeline", tags=["AI Pipeline"])

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-key")

# ── Auth ──────────────────────────────────────────────────────────────────────
# SECURITY CONTRACT (read before adding any route):
#
# This service must NEVER be publicly accessible.
# All requests must come from the backend server, which:
#   1. Validates the user's JWT token
#   2. Replaces user_id in the request body with the real authenticated user ID
#   3. Adds the X-API-KEY header before forwarding here
#
# The API key check below proves the request came from the backend.
# It does NOT prove the user_id in the body is valid — that is the backend's job.

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

class IdeaIntakeInput(BaseModel):
    user_id: str
    message: str
    history: List[Dict[str, Any]] = []

class UserIdInput(BaseModel):
    user_id: str

class SectionChatInput(BaseModel):
    user_id: str
    message: str
    history: List[Dict[str, Any]] = []

class RegenerateCustomInput(BaseModel):
    user_id: str
    custom_prompt: str


# ── SSE streaming helper ──────────────────────────────────────────────────────

def _sse_chat_stream(
    messages: list,
    groq_client: Any,
    groq_model: str,
    temperature: float,
    max_tokens: int,
    on_complete: Callable[[str], Dict[str, Any]],
) -> StreamingResponse:
    """
    Stream LLM tokens as Server-Sent Events and call `on_complete` with the
    full assembled reply once streaming finishes.

    SSE event format
    ────────────────
    Content token:
        data: {"type": "token", "content": "word "}

    Stream finished (includes metadata from on_complete):
        data: {"type": "done", "chat_history_length": 42}

    The caller (route) validates everything before calling this function so
    that HTTP errors are returned normally, not buried inside the SSE stream.

    Args:
        messages:     Full message list passed to the LLM.
        groq_client:  Initialised OpenAI-compatible client.
        groq_model:   Model name string.
        temperature:  Sampling temperature.
        max_tokens:   Max tokens for this call.
        on_complete:  Called once with the full reply string.
                      Must return a dict that is merged into the "done" event.
                      Must save to DB internally (it runs inside the generator).
    """
    def _generator() -> Generator[str, None, None]:
        stream = groq_client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        tokens: list[str] = []

        for chunk in stream:
            delta: str = chunk.choices[0].delta.content or ""
            if delta:
                tokens.append(delta)
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

        # Stream complete — assemble full reply and persist
        full_reply = "".join(tokens)
        metadata   = on_complete(full_reply)
        yield f"data: {json.dumps({'type': 'done', **metadata})}\n\n"

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering so tokens flow immediately
        },
    )

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

    from orchestrator.orchestrator import run_new_user_pipeline
    background_tasks.add_task(run_new_user_pipeline, data.user_id, questionnaire, data.skills)

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
        "intake_ready":    crud.get_idea_intake_json(db, user_id) is not None,
        "idea_ready":      crud.get_idea(db, user_id) is not None,
        "customers_ready":   crud.get_customers_json(db, user_id) is not None,
        "competition_ready":     crud.get_competition_json(db, user_id) is not None,
        "market_potential_ready": crud.get_market_potential_json(db, user_id) is not None,
        "idea_strategy_ready":    crud.get_idea_strategy_json(db, user_id) is not None,
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
    from agents.PipelineRunner import build_context
    stored_questionnaire = questionnaire_row.data
    skills = stored_questionnaire.get("skills", [])

    context = build_context(
        problems_row.data,
        stored_questionnaire,
        skills
    )
    intake = crud.get_idea_intake_json(db, data.user_id)
    if intake:
        context += (
            "\n\n=== USER ORIGINAL IDEA INTAKE ===\n"
            f"{intake}"
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


@router.post("/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_stream(data: ChatInput, db=Depends(get_db)) -> StreamingResponse:
    """
    Streaming version of /chat — returns tokens as Server-Sent Events.

    SSE events:
      data: {"type": "token",  "content": "..."}   ← one per token chunk
      data: {"type": "done",   "chat_history_length": N}  ← stream finished

    The frontend must read the event stream and append tokens to the UI as
    they arrive. The full reply is automatically saved to DB before the
    'done' event fires, so a subsequent GET /idea/{user_id} reflects it.
    """
    run = crud.get_pipeline_status(db, data.user_id)
    if not run or run.status != "done":
        raise HTTPException(status_code=425, detail="Pipeline not ready yet.")

    idea_row          = crud.get_idea(db, data.user_id)
    profile_row       = crud.get_profile(db, data.user_id)
    problems_row      = crud.get_problems(db, data.user_id)
    questionnaire_row = crud.get_questionnaire_output(db, data.user_id)

    if not (idea_row and profile_row and problems_row and questionnaire_row):
        raise HTTPException(status_code=425, detail="Missing required context in DB to chat.")

    from agents.PipelineRunner import build_context, groq_client, GROQ_MODEL, IDEA_SYSTEM_PROMPT

    stored_questionnaire = questionnaire_row.data
    context = build_context(
        problems_row.data,
        stored_questionnaire,
        stored_questionnaire.get("skills", []),
    )
    intake = crud.get_idea_intake_json(db, data.user_id)
    if intake:
        context += f"\n\n=== USER ORIGINAL IDEA INTAKE ===\n{intake}"

    existing_history: list = idea_row.chat_history or []
    messages = [
        {"role": "system", "content": IDEA_SYSTEM_PROMPT},
        {"role": "system", "content": context},
        *existing_history[-20:],
        {"role": "user",   "content": data.message},
    ]

    # Capture variables needed inside the closure
    user_id          = data.user_id
    current_idea_txt = idea_row.current_idea

    def _on_complete(full_reply: str) -> Dict[str, Any]:
        new_idea = full_reply if "💡 IDEA:" in full_reply else current_idea_txt
        updated  = existing_history + [
            {"role": "user",      "content": data.message},
            {"role": "assistant", "content": full_reply},
        ]
        # Use a fresh session — the route's session closes before the generator runs
        with SessionLocal() as s:
            crud.save_idea(s, user_id, new_idea, updated)
        return {"chat_history_length": len(updated)}

    return _sse_chat_stream(
        messages=messages,
        groq_client=groq_client,
        groq_model=GROQ_MODEL,
        temperature=0.4,
        max_tokens=1000,
        on_complete=_on_complete,
    )


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


# ── Returning-user flow: Idea Intake ─────────────────────────────────────────

@router.post("/idea-intake", dependencies=[Depends(verify_api_key)])
def idea_intake(data: IdeaIntakeInput):
    """
    Returning-user flow — step 1.
    POST the user's raw idea (and optionally a clarification history).
    Returns structured intake data when clear, or asks clarifying questions.
    """
    from agents.ThreeIdeaIntakeAgent import run_idea_intake

    result = run_idea_intake(
        user_id=data.user_id,
        user_message=data.message,
        history=data.history,
    )

    return {"user_id": data.user_id, **result}


@router.post("/idea-intake/run-problems/{user_id}", dependencies=[Depends(verify_api_key)])
def idea_intake_run_problems(user_id: str, background_tasks: BackgroundTasks, db=Depends(get_db)):
    """
    Returning-user flow — step 2.
    Call this after idea-intake returns status=ready.
    Runs ProblemDiscovery in the background using the structured intake.
    """
    intake = crud.get_idea_intake_json(db, user_id)
    if not intake or intake.get("_status") == "pending_clarification":
        raise HTTPException(status_code=425, detail="Idea intake not ready. Complete /idea-intake first.")

    crud.upsert_pipeline_status(db, user_id, "pending")

    from orchestrator.orchestrator import run_returning_user_pipeline
    background_tasks.add_task(run_returning_user_pipeline, user_id, intake)

    return JSONResponse(status_code=202, content={
        "user_id": user_id,
        "status": "pending",
        "message": "Problem discovery started. Poll /pipeline/status/{user_id} for progress.",
        "poll_url": f"/pipeline/status/{user_id}",
    })


@router.post("/idea-intake/start-chat", dependencies=[Depends(verify_api_key)])
def idea_intake_start_chat(data: UserIdInput, db=Depends(get_db)):
    """
    Returning-user flow - step 3.
    Bridges IdeaIntake + ProblemDiscovery into the existing idea chat.
    Saves compatibility profile/questionnaire rows so /pipeline/chat can continue normally.
    """
    intake = crud.get_idea_intake_json(db, data.user_id)
    problems_row = crud.get_problems(db, data.user_id)

    if not intake or intake.get("_status") == "pending_clarification":
        raise HTTPException(status_code=425, detail="Idea intake is not ready yet.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problem discovery is not ready yet.")

    from agents.ThreeIdeaIntakeAgent import (
        _build_profile_for_problem_discovery,
        _build_questionnaire_for_problem_discovery,
    )
    from agents.PipelineRunner import build_context, groq_client, GROQ_MODEL, IDEA_SYSTEM_PROMPT

    profile_compat = _build_profile_for_problem_discovery(intake)
    questionnaire_compat = _build_questionnaire_for_problem_discovery(intake)
    questionnaire_compat["skills"] = []

    crud.save_profile(db, data.user_id, profile_compat)
    crud.save_questionnaire_output(db, data.user_id, questionnaire_compat)

    context = build_context(
        problems_row.data,
        questionnaire_compat,
        [],
    )
    context += (
        "\n\n=== USER ORIGINAL IDEA INTAKE ===\n"
        f"Idea summary: {intake.get('idea_summary', '')}\n"
        f"Target users: {intake.get('target_users', [])}\n"
        f"Industry: {intake.get('industry', '')}\n"
        f"Problem assumption: {intake.get('problem_assumption', '')}\n"
        f"Solution assumption: {intake.get('solution_assumption', '')}\n"
        f"Business model: {intake.get('business_model', '')}\n"
        f"Region: {intake.get('region', 'Global')}\n"
    )

    opening = (
        "The user already has this startup idea. Do not generate a random new idea. "
        "Use the original idea intake and discovered problems to refine it into the exact idea format. "
        "Point out the strongest real problem to solve and how the user should adjust the idea."
    )

    messages = [
        {"role": "system", "content": IDEA_SYSTEM_PROMPT},
        {"role": "system", "content": context},
        {"role": "user", "content": opening},
    ]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1000,
    )
    reply = response.choices[0].message.content.strip()

    history = [
        {"role": "user", "content": opening},
        {"role": "assistant", "content": reply},
    ]
    crud.save_idea(db, data.user_id, reply, history)
    crud.upsert_pipeline_status(db, data.user_id, "done", None)

    return {
        "user_id": data.user_id,
        "status": "done",
        "current_idea": reply,
        "chat_history": history,
        "message": "Idea chat is ready. Continue with /pipeline/chat.",
    }


@router.get("/idea-intake/{user_id}", dependencies=[Depends(verify_api_key)])
def get_idea_intake(user_id: str, db=Depends(get_db)):
    """Get the saved idea intake for a returning user."""
    intake = crud.get_idea_intake_json(db, user_id)
    if not intake:
        raise HTTPException(status_code=404, detail="No idea intake found for this user.")

    return {"user_id": user_id, "intake": intake}


# ── Section 4: Customer Analysis ─────────────────────────────────────────────

@router.post("/customers/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_customers(user_id: str, db=Depends(get_db)):
    """
    Generate the customer analysis for the saved idea.
    Requires: idea_ready = true (call /pipeline/run or /idea-intake first).
    """
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)

    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problems not ready. Complete the pipeline first.")

    profile_row = crud.get_profile(db, user_id)
    profile     = profile_row.data if profile_row else None

    from agents.FourCustomersAgent import run_customers_analysis
    result = run_customers_analysis(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        profile=profile,
    )

    return {
        "user_id": user_id,
        "status":  "done",
        "customers": result,
    }


@router.post("/customers/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_customers(user_id: str, db=Depends(get_db)):
    """Regenerate the customer analysis with the same inputs."""
    return generate_customers(user_id, db)


@router.post("/customers/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_customers_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    """Regenerate the customer analysis with an additional custom instruction."""
    idea_row     = crud.get_idea(db, data.user_id)
    problems_row = crud.get_problems(db, data.user_id)

    if not idea_row or not problems_row:
        raise HTTPException(status_code=425, detail="Idea or problems not ready.")

    profile_row = crud.get_profile(db, data.user_id)
    profile     = profile_row.data if profile_row else None

    from agents.FourCustomersAgent import run_customers_analysis
    result = run_customers_analysis(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        profile=profile,
        custom_prompt=data.custom_prompt,
    )

    return {
        "user_id": data.user_id,
        "status":  "done",
        "customers": result,
    }


@router.post("/customers/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_customers(data: SectionChatInput, db=Depends(get_db)):
    """
    Refine the customer analysis through a section-scoped chat.
    Only modifies the customers section — no other pipeline data is touched.
    """
    customers_row = crud.get_customers(db, data.user_id)
    if not customers_row:
        raise HTTPException(
            status_code=425,
            detail="Customer analysis not generated yet. Call POST /customers/{user_id} first."
        )

    from agents.FourCustomersAgent import chat_customers as _chat
    reply = _chat(
        current_analysis=customers_row.data,
        user_message=data.message,
        history=data.history,
    )

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_customers(db, data.user_id, customers_row.data, updated_history)

    return {
        "user_id": data.user_id,
        "reply":   reply,
        "chat_history_length": len(updated_history),
    }


@router.post("/customers/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_customers_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    """
    Streaming version of /customers/{user_id}/chat.
    Returns customer-analysis refinement tokens as Server-Sent Events.
    """
    customers_row = crud.get_customers(db, data.user_id)
    if not customers_row:
        raise HTTPException(status_code=425, detail="Customer analysis not generated yet.")

    from agents.PipelineRunner import groq_client, GROQ_MODEL

    current_data = customers_row.data
    user_id      = data.user_id

    # Build the same messages the non-streaming endpoint would build
    import json as _json
    from System_Messages.customers_prompt import CUSTOMERS_CHAT_PROMPT

    context  = "=== CURRENT CUSTOMER ANALYSIS ===\n" + _json.dumps(current_data, indent=2)
    messages = [
        {"role": "system", "content": CUSTOMERS_CHAT_PROMPT},
        {"role": "system", "content": context},
        *data.history[-20:],
        {"role": "user",   "content": data.message},
    ]

    def _on_complete(full_reply: str) -> Dict[str, Any]:
        updated = data.history + [
            {"role": "user",      "content": data.message},
            {"role": "assistant", "content": full_reply},
        ]
        with SessionLocal() as s:
            crud.save_customers(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return _sse_chat_stream(
        messages=messages,
        groq_client=groq_client,
        groq_model=GROQ_MODEL,
        temperature=0.4,
        max_tokens=1200,
        on_complete=_on_complete,
    )


@router.get("/customers/{user_id}", dependencies=[Depends(verify_api_key)])
def get_customers(user_id: str, db=Depends(get_db)):
    """Get the saved customer analysis and chat history."""
    row = crud.get_customers(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No customer analysis found for this user.")

    return {
        "user_id":      user_id,
        "customers":    row.data,
        "chat_history": row.chat_history or [],
    }


# ── Section 5: Competition Analysis ──────────────────────────────────────────

@router.post("/competition/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_competition(user_id: str, db=Depends(get_db)):
    """
    Generate competition analysis.
    Requires idea_ready = true. Enriches output with customer data if available.
    """
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)

    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problems not ready. Complete the pipeline first.")

    customers_row = crud.get_customers(db, user_id)
    customers     = customers_row.data if customers_row else None

    from agents.FiveCompetitionAgent import run_competition_analysis
    result = run_competition_analysis(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=customers,
    )

    return {"user_id": user_id, "status": "done", "competition": result}


@router.post("/competition/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_competition(user_id: str, db=Depends(get_db)):
    """Regenerate competition analysis with the same inputs."""
    return generate_competition(user_id, db)


@router.post("/competition/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_competition_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    """Regenerate competition analysis with an additional custom instruction."""
    idea_row     = crud.get_idea(db, data.user_id)
    problems_row = crud.get_problems(db, data.user_id)

    if not idea_row or not problems_row:
        raise HTTPException(status_code=425, detail="Idea or problems not ready.")

    customers_row = crud.get_customers(db, data.user_id)
    customers     = customers_row.data if customers_row else None

    from agents.FiveCompetitionAgent import run_competition_analysis
    result = run_competition_analysis(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=customers,
        custom_prompt=data.custom_prompt,
    )

    return {"user_id": data.user_id, "status": "done", "competition": result}


@router.post("/competition/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_competition(data: SectionChatInput, db=Depends(get_db)):
    """
    Refine the competition analysis through a section-scoped chat.
    Only modifies the competition section — no other pipeline data is touched.
    """
    competition_row = crud.get_competition(db, data.user_id)
    if not competition_row:
        raise HTTPException(
            status_code=425,
            detail="Competition analysis not generated yet. Call POST /competition/{user_id} first."
        )

    from agents.FiveCompetitionAgent import chat_competition as _chat
    reply = _chat(
        current_analysis=competition_row.data,
        user_message=data.message,
        history=data.history,
    )

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_competition(db, data.user_id, competition_row.data, updated_history)

    return {
        "user_id": data.user_id,
        "reply":   reply,
        "chat_history_length": len(updated_history),
    }


@router.post("/competition/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_competition_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    """Streaming version of /competition/{user_id}/chat — tokens via SSE."""
    competition_row = crud.get_competition(db, data.user_id)
    if not competition_row:
        raise HTTPException(status_code=425, detail="Competition analysis not generated yet.")

    import json as _json
    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.competition_prompt import COMPETITION_CHAT_PROMPT

    current_data = competition_row.data
    user_id      = data.user_id
    context      = "=== CURRENT COMPETITION ANALYSIS ===\n" + _json.dumps(current_data, indent=2)
    messages     = [
        {"role": "system", "content": COMPETITION_CHAT_PROMPT},
        {"role": "system", "content": context},
        *data.history[-20:],
        {"role": "user",   "content": data.message},
    ]

    def _on_complete(full_reply: str) -> Dict[str, Any]:
        updated = data.history + [
            {"role": "user",      "content": data.message},
            {"role": "assistant", "content": full_reply},
        ]
        with SessionLocal() as s:
            crud.save_competition(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return _sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=1200, on_complete=_on_complete,
    )


@router.get("/competition/{user_id}", dependencies=[Depends(verify_api_key)])
def get_competition(user_id: str, db=Depends(get_db)):
    """Get the saved competition analysis and chat history."""
    row = crud.get_competition(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No competition analysis found for this user.")

    return {
        "user_id":      user_id,
        "competition":  row.data,
        "chat_history": row.chat_history or [],
    }


# ── Section 6: Market Potential ───────────────────────────────────────────────

@router.post("/market-potential/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_market_potential(user_id: str, db=Depends(get_db)):
    """
    Generate market potential analysis (TAM/SAM/SOM + PESTEL).
    Enriched with web-sourced market data when available.
    Requires idea_ready = true. Pulls customers + competition if available.
    """
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)

    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problems not ready. Complete the pipeline first.")

    customers_row   = crud.get_customers(db, user_id)
    competition_row = crud.get_competition(db, user_id)

    from agents.SixMaketPotential import run_market_potential
    result = run_market_potential(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=customers_row.data if customers_row else None,
        competition=competition_row.data if competition_row else None,
    )

    return {"user_id": user_id, "status": "done", "market_potential": result}


@router.post("/market-potential/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_market_potential(user_id: str, db=Depends(get_db)):
    """Regenerate market potential analysis with the same inputs."""
    return generate_market_potential(user_id, db)


@router.post("/market-potential/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_market_potential_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    """Regenerate market potential with an additional custom instruction."""
    idea_row     = crud.get_idea(db, data.user_id)
    problems_row = crud.get_problems(db, data.user_id)

    if not idea_row or not problems_row:
        raise HTTPException(status_code=425, detail="Idea or problems not ready.")

    customers_row   = crud.get_customers(db, data.user_id)
    competition_row = crud.get_competition(db, data.user_id)

    from agents.SixMaketPotential import run_market_potential
    result = run_market_potential(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=customers_row.data if customers_row else None,
        competition=competition_row.data if competition_row else None,
        custom_prompt=data.custom_prompt,
    )

    return {"user_id": data.user_id, "status": "done", "market_potential": result}


@router.post("/market-potential/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_market_potential(data: SectionChatInput, db=Depends(get_db)):
    """
    Refine the market potential analysis through a section-scoped chat.
    Only modifies market potential — no other pipeline data is touched.
    """
    mp_row = crud.get_market_potential(db, data.user_id)
    if not mp_row:
        raise HTTPException(
            status_code=425,
            detail="Market potential not generated yet. Call POST /market-potential/{user_id} first."
        )

    from agents.SixMaketPotential import chat_market_potential as _chat
    reply = _chat(
        current_analysis=mp_row.data,
        user_message=data.message,
        history=data.history,
    )

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_market_potential(db, data.user_id, mp_row.data, updated_history)

    return {
        "user_id": data.user_id,
        "reply":   reply,
        "chat_history_length": len(updated_history),
    }


@router.post("/market-potential/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_market_potential_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    """Streaming version of /market-potential/{user_id}/chat — tokens via SSE."""
    mp_row = crud.get_market_potential(db, data.user_id)
    if not mp_row:
        raise HTTPException(status_code=425, detail="Market potential not generated yet.")

    import json as _json
    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.market_potential_prompt import MARKET_POTENTIAL_CHAT_PROMPT

    current_data = mp_row.data
    user_id      = data.user_id
    context      = "=== CURRENT MARKET POTENTIAL ANALYSIS ===\n" + _json.dumps(current_data, indent=2)
    messages     = [
        {"role": "system", "content": MARKET_POTENTIAL_CHAT_PROMPT},
        {"role": "system", "content": context},
        *data.history[-20:],
        {"role": "user",   "content": data.message},
    ]

    def _on_complete(full_reply: str) -> Dict[str, Any]:
        updated = data.history + [
            {"role": "user",      "content": data.message},
            {"role": "assistant", "content": full_reply},
        ]
        with SessionLocal() as s:
            crud.save_market_potential(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return _sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=1200, on_complete=_on_complete,
    )


@router.get("/market-potential/{user_id}", dependencies=[Depends(verify_api_key)])
def get_market_potential(user_id: str, db=Depends(get_db)):
    """Get the saved market potential analysis and chat history."""
    row = crud.get_market_potential(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No market potential analysis found for this user.")

    return {
        "user_id":          user_id,
        "market_potential": row.data,
        "chat_history":     row.chat_history or [],
    }


# ── Section 7: Idea Strategy ──────────────────────────────────────────────────

@router.post("/idea-strategy/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_idea_strategy(user_id: str, db=Depends(get_db)):
    """
    Generate the idea strategy (value prop, positioning, assumptions, validation plan).
    Pulls all available prior sections from DB to enrich the analysis.
    Requires idea_ready = true.
    """
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)

    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problems not ready. Complete the pipeline first.")

    customers_row   = crud.get_customers(db, user_id)
    competition_row = crud.get_competition(db, user_id)
    mp_row          = crud.get_market_potential(db, user_id)

    from agents.SevenIdeaStrategy import run_idea_strategy
    result = run_idea_strategy(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=customers_row.data   if customers_row   else None,
        competition=competition_row.data if competition_row else None,
        market_potential=mp_row.data   if mp_row          else None,
    )

    return {"user_id": user_id, "status": "done", "idea_strategy": result}


@router.post("/idea-strategy/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_idea_strategy(user_id: str, db=Depends(get_db)):
    """Regenerate the idea strategy with the same inputs."""
    return generate_idea_strategy(user_id, db)


@router.post("/idea-strategy/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_idea_strategy_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    """Regenerate the idea strategy with an additional custom instruction."""
    idea_row     = crud.get_idea(db, data.user_id)
    problems_row = crud.get_problems(db, data.user_id)

    if not idea_row or not problems_row:
        raise HTTPException(status_code=425, detail="Idea or problems not ready.")

    customers_row   = crud.get_customers(db, data.user_id)
    competition_row = crud.get_competition(db, data.user_id)
    mp_row          = crud.get_market_potential(db, data.user_id)

    from agents.SevenIdeaStrategy import run_idea_strategy
    result = run_idea_strategy(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=customers_row.data   if customers_row   else None,
        competition=competition_row.data if competition_row else None,
        market_potential=mp_row.data   if mp_row          else None,
        custom_prompt=data.custom_prompt,
    )

    return {"user_id": data.user_id, "status": "done", "idea_strategy": result}


@router.post("/idea-strategy/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_idea_strategy(data: SectionChatInput, db=Depends(get_db)):
    """
    Refine the idea strategy through a section-scoped chat.
    Only modifies strategy — no other pipeline data is touched.
    """
    strategy_row = crud.get_idea_strategy(db, data.user_id)
    if not strategy_row:
        raise HTTPException(
            status_code=425,
            detail="Idea strategy not generated yet. Call POST /idea-strategy/{user_id} first."
        )

    from agents.SevenIdeaStrategy import chat_idea_strategy as _chat
    reply = _chat(
        current_analysis=strategy_row.data,
        user_message=data.message,
        history=data.history,
    )

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_idea_strategy(db, data.user_id, strategy_row.data, updated_history)

    return {
        "user_id": data.user_id,
        "reply":   reply,
        "chat_history_length": len(updated_history),
    }


@router.post("/idea-strategy/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_idea_strategy_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    """Streaming version of /idea-strategy/{user_id}/chat — tokens via SSE."""
    strategy_row = crud.get_idea_strategy(db, data.user_id)
    if not strategy_row:
        raise HTTPException(status_code=425, detail="Idea strategy not generated yet.")

    import json as _json
    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.idea_strategy_prompt import IDEA_STRATEGY_CHAT_PROMPT

    current_data = strategy_row.data
    user_id      = data.user_id
    context      = "=== CURRENT IDEA STRATEGY ===\n" + _json.dumps(current_data, indent=2)
    messages     = [
        {"role": "system", "content": IDEA_STRATEGY_CHAT_PROMPT},
        {"role": "system", "content": context},
        *data.history[-20:],
        {"role": "user",   "content": data.message},
    ]

    def _on_complete(full_reply: str) -> Dict[str, Any]:
        updated = data.history + [
            {"role": "user",      "content": data.message},
            {"role": "assistant", "content": full_reply},
        ]
        with SessionLocal() as s:
            crud.save_idea_strategy(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return _sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=1200, on_complete=_on_complete,
    )


@router.get("/idea-strategy/{user_id}", dependencies=[Depends(verify_api_key)])
def get_idea_strategy(user_id: str, db=Depends(get_db)):
    """Get the saved idea strategy and chat history."""
    row = crud.get_idea_strategy(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No idea strategy found for this user.")

    return {
        "user_id":       user_id,
        "idea_strategy": row.data,
        "chat_history":  row.chat_history or [],
    }
