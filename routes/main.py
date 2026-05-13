import json
import time
from collections.abc import Generator
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from db.connection import get_db, SessionLocal
from db import crud


router = APIRouter(prefix="/pipeline", tags=["AI Pipeline"])

from agents.config import API_SECRET_KEY

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

class ChatInput(BaseModel):
    user_id: str
    message: str

class GeneralBotInput(BaseModel):
    user_id: str
    message: str
    history: List[Dict[str, Any]] = []

class ExplainerInput(BaseModel):
    user_id: str
    message: str
    history: List[Dict[str, Any]] = []
    section: Optional[str] = None   # e.g. "customers", "business_model"

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
    questionnaire = crud.get_questionnaire_from_profile(db, data.user_id)
    if not questionnaire:
        raise HTTPException(
            status_code=425,
            detail="user_profiles row not found or questionnaire_json is empty. "
                   "Complete the onboarding questionnaire first.",
        )

    skills = crud.get_skills_from_profile(db, data.user_id)

    crud.save_questionnaire_output(db, data.user_id, questionnaire)
    crud.upsert_pipeline_status(db, data.user_id, "pending")

    from orchestrator.orchestrator import run_new_user_pipeline
    background_tasks.add_task(run_new_user_pipeline, data.user_id, questionnaire, skills)

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

    # Query each section once and reuse for both individual flags and pipeline_complete
    profile_ready         = crud.get_profile(db, user_id) is not None
    problems_ready        = crud.get_problems(db, user_id) is not None
    intake_ready          = crud.get_idea_intake_json(db, user_id) is not None
    idea_ready            = crud.get_idea(db, user_id) is not None
    customers_ready       = crud.get_customers_json(db, user_id) is not None
    competition_ready     = crud.get_competition_json(db, user_id) is not None
    market_potential_ready = crud.get_market_potential_json(db, user_id) is not None
    idea_strategy_ready   = crud.get_idea_strategy_json(db, user_id) is not None
    business_model_ready  = crud.get_business_model_json(db, user_id) is not None
    functions_list_ready  = crud.get_functions_list_json(db, user_id) is not None
    mvp_planning_ready    = crud.get_mvp_planning_json(db, user_id) is not None
    unit_economics_ready  = crud.get_unit_economics_json(db, user_id) is not None
    go_to_market_ready    = crud.get_go_to_market_json(db, user_id) is not None

    # pipeline_complete = all 12 analysis sections are done (true "full plan" flag)
    pipeline_complete = all([
        profile_ready, problems_ready, idea_ready,
        customers_ready, competition_ready, market_potential_ready,
        idea_strategy_ready, business_model_ready, functions_list_ready,
        mvp_planning_ready, unit_economics_ready, go_to_market_ready,
    ])

    return {
        "user_id":                user_id,
        "status":                 run.status,
        "current_step":           run.current_step,
        # Automatic pipeline (steps 1-3)
        "profile_ready":          profile_ready,
        "problems_ready":         problems_ready,
        "intake_ready":           intake_ready,
        "idea_ready":             idea_ready,
        # User-triggered analysis sections (steps 4-12)
        "customers_ready":        customers_ready,
        "competition_ready":      competition_ready,
        "market_potential_ready": market_potential_ready,
        "idea_strategy_ready":    idea_strategy_ready,
        "business_model_ready":   business_model_ready,
        "functions_list_ready":   functions_list_ready,
        "mvp_planning_ready":     mvp_planning_ready,
        "unit_economics_ready":   unit_economics_ready,
        "go_to_market_ready":     go_to_market_ready,
        # True only when ALL 12 sections are complete
        "pipeline_complete":      pipeline_complete,
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


# ── Section 8: Business Model ─────────────────────────────────────────────────

def _load_business_model_deps(user_id: str, db: Any) -> tuple:
    """Pull all inputs the business model agent needs from DB in one place."""
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)
    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problems not ready. Complete the pipeline first.")
    return (
        idea_row,
        problems_row,
        crud.get_customers(db, user_id),
        crud.get_competition(db, user_id),
        crud.get_market_potential(db, user_id),
        crud.get_idea_strategy(db, user_id),
    )


@router.post("/business-model/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_business_model(user_id: str, db=Depends(get_db)):
    """
    Generate the business model (BMC, revenue streams, pricing, cost structure).
    Pulls all available prior sections from DB to enrich the output.
    Requires idea_ready = true.
    """
    idea_row, problems_row, cust_row, comp_row, mp_row, strat_row = \
        _load_business_model_deps(user_id, db)

    from agents.EightBusinessModel import run_business_model
    result = run_business_model(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=cust_row.data    if cust_row  else None,
        competition=comp_row.data  if comp_row  else None,
        market_potential=mp_row.data if mp_row  else None,
        strategy=strat_row.data    if strat_row else None,
    )

    return {"user_id": user_id, "status": "done", "business_model": result}


@router.post("/business-model/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_business_model(user_id: str, db=Depends(get_db)):
    """Regenerate the business model with the same inputs."""
    return generate_business_model(user_id, db)


@router.post("/business-model/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_business_model_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    """Regenerate the business model with an additional custom instruction."""
    idea_row, problems_row, cust_row, comp_row, mp_row, strat_row = \
        _load_business_model_deps(data.user_id, db)

    from agents.EightBusinessModel import run_business_model
    result = run_business_model(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=cust_row.data    if cust_row  else None,
        competition=comp_row.data  if comp_row  else None,
        market_potential=mp_row.data if mp_row  else None,
        strategy=strat_row.data    if strat_row else None,
        custom_prompt=data.custom_prompt,
    )

    return {"user_id": data.user_id, "status": "done", "business_model": result}


@router.post("/business-model/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_business_model(data: SectionChatInput, db=Depends(get_db)):
    """
    Refine the business model through a section-scoped chat.
    Only modifies the business model — no other pipeline data is touched.
    """
    bm_row = crud.get_business_model(db, data.user_id)
    if not bm_row:
        raise HTTPException(
            status_code=425,
            detail="Business model not generated yet. Call POST /business-model/{user_id} first."
        )

    from agents.EightBusinessModel import chat_business_model as _chat
    reply = _chat(
        current_analysis=bm_row.data,
        user_message=data.message,
        history=data.history,
    )

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_business_model(db, data.user_id, bm_row.data, updated_history)

    return {
        "user_id": data.user_id,
        "reply":   reply,
        "chat_history_length": len(updated_history),
    }


@router.post("/business-model/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_business_model_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    """Streaming version of /business-model/{user_id}/chat — tokens via SSE."""
    bm_row = crud.get_business_model(db, data.user_id)
    if not bm_row:
        raise HTTPException(status_code=425, detail="Business model not generated yet.")

    import json as _json
    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.business_model_prompt import BUSINESS_MODEL_CHAT_PROMPT

    current_data = bm_row.data
    user_id      = data.user_id
    context      = "=== CURRENT BUSINESS MODEL ===\n" + _json.dumps(current_data, indent=2)
    messages     = [
        {"role": "system", "content": BUSINESS_MODEL_CHAT_PROMPT},
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
            crud.save_business_model(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return _sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=1200, on_complete=_on_complete,
    )


@router.get("/business-model/{user_id}", dependencies=[Depends(verify_api_key)])
def get_business_model(user_id: str, db=Depends(get_db)):
    """Get the saved business model and chat history."""
    row = crud.get_business_model(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No business model found for this user.")

    return {
        "user_id":        user_id,
        "business_model": row.data,
        "chat_history":   row.chat_history or [],
    }


# ── Section 9: Product Functions List ────────────────────────────────────────

def _load_functions_list_deps(user_id: str, db: Any) -> tuple:
    """Pull all inputs the functions list agent needs from DB."""
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)
    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problems not ready. Complete the pipeline first.")
    return (
        idea_row,
        problems_row,
        crud.get_customers(db, user_id),
        crud.get_competition(db, user_id),
        crud.get_market_potential(db, user_id),
        crud.get_idea_strategy(db, user_id),
        crud.get_business_model(db, user_id),
    )


@router.post("/functions-list/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_functions_list(user_id: str, db=Depends(get_db)):
    """
    Generate the product functions list (core, nice-to-have, future capabilities,
    feature creep warnings, no-code stack).
    Requires idea_ready = true. Pulls all prior sections if available.
    """
    idea_row, problems_row, cust_row, comp_row, mp_row, strat_row, bm_row = \
        _load_functions_list_deps(user_id, db)

    from agents.NineFunctionsList import run_functions_list
    result = run_functions_list(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=cust_row.data      if cust_row   else None,
        competition=comp_row.data    if comp_row   else None,
        market_potential=mp_row.data if mp_row     else None,
        strategy=strat_row.data      if strat_row  else None,
        business_model=bm_row.data   if bm_row     else None,
    )

    return {"user_id": user_id, "status": "done", "functions_list": result}


@router.post("/functions-list/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_functions_list(user_id: str, db=Depends(get_db)):
    """Regenerate the functions list with the same inputs."""
    return generate_functions_list(user_id, db)


@router.post("/functions-list/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_functions_list_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    """Regenerate the functions list with an additional custom instruction."""
    idea_row, problems_row, cust_row, comp_row, mp_row, strat_row, bm_row = \
        _load_functions_list_deps(data.user_id, db)

    from agents.NineFunctionsList import run_functions_list
    result = run_functions_list(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=cust_row.data      if cust_row   else None,
        competition=comp_row.data    if comp_row   else None,
        market_potential=mp_row.data if mp_row     else None,
        strategy=strat_row.data      if strat_row  else None,
        business_model=bm_row.data   if bm_row     else None,
        custom_prompt=data.custom_prompt,
    )

    return {"user_id": data.user_id, "status": "done", "functions_list": result}


@router.post("/functions-list/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_functions_list(data: SectionChatInput, db=Depends(get_db)):
    """
    Refine the functions list through a section-scoped chat.
    Only modifies the functions list — no other pipeline data is touched.
    """
    fl_row = crud.get_functions_list(db, data.user_id)
    if not fl_row:
        raise HTTPException(
            status_code=425,
            detail="Functions list not generated yet. Call POST /functions-list/{user_id} first."
        )

    from agents.NineFunctionsList import chat_functions_list as _chat
    reply = _chat(
        current_analysis=fl_row.data,
        user_message=data.message,
        history=data.history,
    )

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_functions_list(db, data.user_id, fl_row.data, updated_history)

    return {
        "user_id": data.user_id,
        "reply":   reply,
        "chat_history_length": len(updated_history),
    }


@router.post("/functions-list/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_functions_list_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    """Streaming version of /functions-list/{user_id}/chat — tokens via SSE."""
    fl_row = crud.get_functions_list(db, data.user_id)
    if not fl_row:
        raise HTTPException(status_code=425, detail="Functions list not generated yet.")

    import json as _json
    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.functions_list_prompt import FUNCTIONS_LIST_CHAT_PROMPT

    current_data = fl_row.data
    user_id      = data.user_id
    context      = "=== CURRENT PRODUCT FUNCTIONS LIST ===\n" + _json.dumps(current_data, indent=2)
    messages     = [
        {"role": "system", "content": FUNCTIONS_LIST_CHAT_PROMPT},
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
            crud.save_functions_list(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return _sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=1200, on_complete=_on_complete,
    )


@router.get("/functions-list/{user_id}", dependencies=[Depends(verify_api_key)])
def get_functions_list(user_id: str, db=Depends(get_db)):
    """Get the saved product functions list and chat history."""
    row = crud.get_functions_list(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No functions list found for this user.")

    return {
        "user_id":        user_id,
        "functions_list": row.data,
        "chat_history":   row.chat_history or [],
    }


# ── Section 10: MVP Planning ──────────────────────────────────────────────────

def _load_mvp_planning_deps(user_id: str, db: Any) -> tuple:
    """Pull all inputs the MVP planning agent needs from DB."""
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)
    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problems not ready. Complete the pipeline first.")
    return (
        idea_row,
        problems_row,
        crud.get_customers(db, user_id),
        crud.get_market_potential(db, user_id),
        crud.get_idea_strategy(db, user_id),
        crud.get_business_model(db, user_id),
        crud.get_functions_list(db, user_id),
    )


@router.post("/mvp-planning/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_mvp_planning(user_id: str, db=Depends(get_db)):
    """
    Generate the MVP plan: goal, riskiest assumptions, scope, user flows,
    build phases, validation experiments, launch criteria, QA checklist.
    Requires idea_ready = true. Pulls all prior sections if available.
    """
    idea_row, problems_row, cust_row, mp_row, strat_row, bm_row, fl_row = \
        _load_mvp_planning_deps(user_id, db)

    from agents.TenMVPPlanning import run_mvp_planning
    result = run_mvp_planning(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=cust_row.data      if cust_row   else None,
        market_potential=mp_row.data if mp_row     else None,
        strategy=strat_row.data      if strat_row  else None,
        business_model=bm_row.data   if bm_row     else None,
        functions_list=fl_row.data   if fl_row     else None,
    )

    return {"user_id": user_id, "status": "done", "mvp_planning": result}


@router.post("/mvp-planning/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_mvp_planning(user_id: str, db=Depends(get_db)):
    """Regenerate the MVP plan with the same inputs."""
    return generate_mvp_planning(user_id, db)


@router.post("/mvp-planning/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_mvp_planning_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    """Regenerate the MVP plan with an additional custom instruction."""
    idea_row, problems_row, cust_row, mp_row, strat_row, bm_row, fl_row = \
        _load_mvp_planning_deps(data.user_id, db)

    from agents.TenMVPPlanning import run_mvp_planning
    result = run_mvp_planning(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        problems=problems_row.data,
        customers=cust_row.data      if cust_row   else None,
        market_potential=mp_row.data if mp_row     else None,
        strategy=strat_row.data      if strat_row  else None,
        business_model=bm_row.data   if bm_row     else None,
        functions_list=fl_row.data   if fl_row     else None,
        custom_prompt=data.custom_prompt,
    )

    return {"user_id": data.user_id, "status": "done", "mvp_planning": result}


@router.post("/mvp-planning/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_mvp_planning(data: SectionChatInput, db=Depends(get_db)):
    """
    Refine the MVP plan through a section-scoped chat.
    Only modifies MVP planning — no other pipeline data is touched.
    """
    mvp_row = crud.get_mvp_planning(db, data.user_id)
    if not mvp_row:
        raise HTTPException(
            status_code=425,
            detail="MVP plan not generated yet. Call POST /mvp-planning/{user_id} first."
        )

    from agents.TenMVPPlanning import chat_mvp_planning as _chat
    reply = _chat(
        current_analysis=mvp_row.data,
        user_message=data.message,
        history=data.history,
    )

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_mvp_planning(db, data.user_id, mvp_row.data, updated_history)

    return {
        "user_id": data.user_id,
        "reply":   reply,
        "chat_history_length": len(updated_history),
    }


@router.post("/mvp-planning/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_mvp_planning_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    """Streaming version of /mvp-planning/{user_id}/chat — tokens via SSE."""
    mvp_row = crud.get_mvp_planning(db, data.user_id)
    if not mvp_row:
        raise HTTPException(status_code=425, detail="MVP plan not generated yet.")

    import json as _json
    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.mvp_planning_prompt import MVP_PLANNING_CHAT_PROMPT

    current_data = mvp_row.data
    user_id      = data.user_id
    context      = "=== CURRENT MVP PLAN ===\n" + _json.dumps(current_data, indent=2)
    messages     = [
        {"role": "system", "content": MVP_PLANNING_CHAT_PROMPT},
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
            crud.save_mvp_planning(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return _sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=1200, on_complete=_on_complete,
    )


@router.get("/mvp-planning/{user_id}", dependencies=[Depends(verify_api_key)])
def get_mvp_planning(user_id: str, db=Depends(get_db)):
    """Get the saved MVP plan and chat history."""
    row = crud.get_mvp_planning(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No MVP plan found for this user.")

    return {
        "user_id":      user_id,
        "mvp_planning": row.data,
        "chat_history": row.chat_history or [],
    }


# ── Section 11: Unit Economics ────────────────────────────────────────────────

def _load_unit_economics_deps(user_id: str, db: Any) -> tuple:
    """Pull all inputs the unit economics agent needs from DB."""
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)
    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problems not ready. Complete the pipeline first.")
    return (
        idea_row,
        crud.get_customers(db, user_id),
        crud.get_market_potential(db, user_id),
        crud.get_idea_strategy(db, user_id),
        crud.get_business_model(db, user_id),
        crud.get_mvp_planning(db, user_id),
    )


@router.post("/unit-economics/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_unit_economics(user_id: str, db=Depends(get_db)):
    """
    Generate unit economics: CAC, LTV, LTV/CAC, payback period,
    break-even, monthly projections, pricing tests, viability verdict.
    Requires idea_ready = true. Uses business_model as primary input.
    """
    idea_row, cust_row, mp_row, strat_row, bm_row, mvp_row = \
        _load_unit_economics_deps(user_id, db)

    from agents.ElevenUnitEconomicsAgent import run_unit_economics
    result = run_unit_economics(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        customers=cust_row.data      if cust_row   else None,
        market_potential=mp_row.data if mp_row     else None,
        strategy=strat_row.data      if strat_row  else None,
        business_model=bm_row.data   if bm_row     else None,
        mvp_planning=mvp_row.data    if mvp_row    else None,
    )

    return {"user_id": user_id, "status": "done", "unit_economics": result}


@router.post("/unit-economics/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_unit_economics(user_id: str, db=Depends(get_db)):
    """Regenerate unit economics with the same inputs."""
    return generate_unit_economics(user_id, db)


@router.post("/unit-economics/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_unit_economics_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    """Regenerate unit economics with an additional custom instruction."""
    idea_row, cust_row, mp_row, strat_row, bm_row, mvp_row = \
        _load_unit_economics_deps(data.user_id, db)

    from agents.ElevenUnitEconomicsAgent import run_unit_economics
    result = run_unit_economics(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        customers=cust_row.data      if cust_row   else None,
        market_potential=mp_row.data if mp_row     else None,
        strategy=strat_row.data      if strat_row  else None,
        business_model=bm_row.data   if bm_row     else None,
        mvp_planning=mvp_row.data    if mvp_row    else None,
        custom_prompt=data.custom_prompt,
    )

    return {"user_id": data.user_id, "status": "done", "unit_economics": result}


@router.post("/unit-economics/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_unit_economics(data: SectionChatInput, db=Depends(get_db)):
    """
    Refine unit economics through a section-scoped chat.
    Supports: stress-testing assumptions, recalculating with new inputs,
    scenario planning (best / base / worst case).
    """
    ue_row = crud.get_unit_economics(db, data.user_id)
    if not ue_row:
        raise HTTPException(
            status_code=425,
            detail="Unit economics not generated yet. Call POST /unit-economics/{user_id} first."
        )

    from agents.ElevenUnitEconomicsAgent import chat_unit_economics as _chat
    reply = _chat(
        current_analysis=ue_row.data,
        user_message=data.message,
        history=data.history,
    )

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_unit_economics(db, data.user_id, ue_row.data, updated_history)

    return {
        "user_id": data.user_id,
        "reply":   reply,
        "chat_history_length": len(updated_history),
    }


@router.post("/unit-economics/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_unit_economics_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    """Streaming version of /unit-economics/{user_id}/chat — tokens via SSE."""
    ue_row = crud.get_unit_economics(db, data.user_id)
    if not ue_row:
        raise HTTPException(status_code=425, detail="Unit economics not generated yet.")

    import json as _json
    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.unit_economics_prompt import UNIT_ECONOMICS_CHAT_PROMPT

    current_data = ue_row.data
    user_id      = data.user_id
    context      = "=== CURRENT UNIT ECONOMICS ===\n" + _json.dumps(current_data, indent=2)
    messages     = [
        {"role": "system", "content": UNIT_ECONOMICS_CHAT_PROMPT},
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
            crud.save_unit_economics(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return _sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.3, max_tokens=1400, on_complete=_on_complete,
    )


@router.get("/unit-economics/{user_id}", dependencies=[Depends(verify_api_key)])
def get_unit_economics(user_id: str, db=Depends(get_db)):
    """Get the saved unit economics analysis and chat history."""
    row = crud.get_unit_economics(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No unit economics found for this user.")

    return {
        "user_id":        user_id,
        "unit_economics": row.data,
        "chat_history":   row.chat_history or [],
    }


# ── Section 12: Go-To-Market Plan (Final) ─────────────────────────────────────

def _load_gtm_deps(user_id: str, db: Any) -> tuple:
    """Pull all inputs the GTM agent needs from DB."""
    idea_row     = crud.get_idea(db, user_id)
    problems_row = crud.get_problems(db, user_id)
    if not idea_row:
        raise HTTPException(status_code=425, detail="Idea not ready. Complete the pipeline first.")
    if not problems_row:
        raise HTTPException(status_code=425, detail="Problems not ready. Complete the pipeline first.")
    return (
        idea_row,
        problems_row,
        crud.get_customers(db, user_id),
        crud.get_competition(db, user_id),
        crud.get_market_potential(db, user_id),
        crud.get_idea_strategy(db, user_id),
        crud.get_business_model(db, user_id),
        crud.get_mvp_planning(db, user_id),
        crud.get_unit_economics(db, user_id),
    )


@router.post("/go-to-market/{user_id}", dependencies=[Depends(verify_api_key)])
def generate_go_to_market(user_id: str, db=Depends(get_db)):
    """
    Generate the go-to-market plan: launch segment, positioning, channels,
    funnel, experiments, first 100 customers plan, 8-week timeline, CAC tracking.
    This is the final pipeline step. Pulls all available prior sections.
    Requires idea_ready = true.
    """
    idea_row, prob_row, cust_row, comp_row, mp_row, strat_row, bm_row, mvp_row, ue_row = \
        _load_gtm_deps(user_id, db)

    from agents.TwelveGoToMarket import run_go_to_market
    result = run_go_to_market(
        user_id=user_id,
        idea=idea_row.current_idea or "",
        problems=prob_row.data,
        customers=cust_row.data      if cust_row   else None,
        competition=comp_row.data    if comp_row   else None,
        market_potential=mp_row.data if mp_row     else None,
        strategy=strat_row.data      if strat_row  else None,
        business_model=bm_row.data   if bm_row     else None,
        mvp_planning=mvp_row.data    if mvp_row    else None,
        unit_economics=ue_row.data   if ue_row     else None,
    )

    return {"user_id": user_id, "status": "done", "go_to_market": result}


@router.post("/go-to-market/{user_id}/regenerate", dependencies=[Depends(verify_api_key)])
def regenerate_go_to_market(user_id: str, db=Depends(get_db)):
    """Regenerate the GTM plan with the same inputs."""
    return generate_go_to_market(user_id, db)


@router.post("/go-to-market/{user_id}/regenerate-custom", dependencies=[Depends(verify_api_key)])
def regenerate_go_to_market_custom(data: RegenerateCustomInput, db=Depends(get_db)):
    """Regenerate the GTM plan with an additional custom instruction."""
    idea_row, prob_row, cust_row, comp_row, mp_row, strat_row, bm_row, mvp_row, ue_row = \
        _load_gtm_deps(data.user_id, db)

    from agents.TwelveGoToMarket import run_go_to_market
    result = run_go_to_market(
        user_id=data.user_id,
        idea=idea_row.current_idea or "",
        problems=prob_row.data,
        customers=cust_row.data      if cust_row   else None,
        competition=comp_row.data    if comp_row   else None,
        market_potential=mp_row.data if mp_row     else None,
        strategy=strat_row.data      if strat_row  else None,
        business_model=bm_row.data   if bm_row     else None,
        mvp_planning=mvp_row.data    if mvp_row    else None,
        unit_economics=ue_row.data   if ue_row     else None,
        custom_prompt=data.custom_prompt,
    )

    return {"user_id": data.user_id, "status": "done", "go_to_market": result}


@router.post("/go-to-market/{user_id}/chat", dependencies=[Depends(verify_api_key)])
def chat_go_to_market(data: SectionChatInput, db=Depends(get_db)):
    """
    Refine the GTM plan through a section-scoped chat.
    Supports: channel changes, messaging revisions, experiment redesign,
    timeline adjustments, kill metric tuning.
    """
    gtm_row = crud.get_go_to_market(db, data.user_id)
    if not gtm_row:
        raise HTTPException(
            status_code=425,
            detail="GTM plan not generated yet. Call POST /go-to-market/{user_id} first."
        )

    from agents.TwelveGoToMarket import chat_go_to_market as _chat
    reply = _chat(
        current_analysis=gtm_row.data,
        user_message=data.message,
        history=data.history,
    )

    updated_history = data.history + [
        {"role": "user",      "content": data.message},
        {"role": "assistant", "content": reply},
    ]
    crud.save_go_to_market(db, data.user_id, gtm_row.data, updated_history)

    return {
        "user_id": data.user_id,
        "reply":   reply,
        "chat_history_length": len(updated_history),
    }


@router.post("/go-to-market/{user_id}/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_go_to_market_stream(data: SectionChatInput, db=Depends(get_db)) -> StreamingResponse:
    """Streaming version of /go-to-market/{user_id}/chat — tokens via SSE."""
    gtm_row = crud.get_go_to_market(db, data.user_id)
    if not gtm_row:
        raise HTTPException(status_code=425, detail="GTM plan not generated yet.")

    import json as _json
    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.go_to_market_prompt import GO_TO_MARKET_CHAT_PROMPT

    current_data = gtm_row.data
    user_id      = data.user_id
    context      = "=== CURRENT GO-TO-MARKET PLAN ===\n" + _json.dumps(current_data, indent=2)
    messages     = [
        {"role": "system", "content": GO_TO_MARKET_CHAT_PROMPT},
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
            crud.save_go_to_market(s, user_id, current_data, updated)
        return {"chat_history_length": len(updated)}

    return _sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=1400, on_complete=_on_complete,
    )


@router.get("/go-to-market/{user_id}", dependencies=[Depends(verify_api_key)])
def get_go_to_market(user_id: str, db=Depends(get_db)):
    """Get the saved go-to-market plan and chat history."""
    row = crud.get_go_to_market(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No GTM plan found for this user.")

    return {
        "user_id":      user_id,
        "go_to_market": row.data,
        "chat_history": row.chat_history or [],
    }


# ── General Bot ───────────────────────────────────────────────────────────────

@router.post("/general-chat", dependencies=[Depends(verify_api_key)])
def general_chat(data: GeneralBotInput) -> Dict[str, Any]:
    """
    General-purpose startup planning chatbot.

    Understands natural language, classifies intent, and either:
    - Answers from existing pipeline data in the DB
    - Tells the caller which agent endpoint to trigger next
    - Declines gracefully if the question is out of scope

    Response shape
    ──────────────
    {
      "user_id":          str
      "reply":            str    — the assistant's message to show the user
      "intent":           str    — classified intent (chat_about_data | run_section | …)
      "section":          str | null
      "action":           str    — "answered" | "route_to_section" | "declined" | "status"
      "route_to_trigger": str | null  — e.g. "/pipeline/customers/abc123"
      "trigger_needed":   bool   — True if the backend should POST to route_to_trigger
      "chat_history_length": int
    }

    When trigger_needed = true, the backend proxy should:
      1. Show the user the reply
      2. POST to route_to_trigger with the X-API-KEY header
      3. Return the agent's result to the frontend
    """
    from agents.generalBot import run_general_bot

    result = run_general_bot(
        user_id=data.user_id,
        message=data.message,
        history=data.history,
    )

    return {
        "user_id": data.user_id,
        **result,
        "chat_history_length": len(data.history) + 2,
    }


@router.post("/general-chat/stream", dependencies=[Depends(verify_api_key)])
def general_chat_stream(data: GeneralBotInput) -> StreamingResponse:
    """
    Streaming version of /general-chat for intents that produce a plain text
    answer (chat_about_data, general_startup_chat, pipeline_status).

    For routing intents (run_section, refine_section, start_pipeline, out_of_scope)
    the stream returns a single short message — the actual agent work is done
    by the caller posting to route_to_trigger separately.

    SSE events:
      data: {"type": "token",  "content": "..."}
      data: {"type": "done",   "intent": "...", "action": "...", "route_to_trigger": "..." | null}
    """
    from agents.generalBot import (
        _classify_intent,
        _load_pipeline_snapshot,
        _load_section_data,
        _respond_from_data,
        _respond_pipeline_status,
        _respond_run_section,
        _respond_refine_section,
        OUT_OF_SCOPE_RESPONSE,
    )
    from db.connection import SessionLocal as _SL
    from agents.PipelineRunner import groq_client, GROQ_MODEL
    from System_Messages.general_bot_prompt import GENERAL_BOT_SYSTEM_PROMPT
    import json as _json

    # All validation and intent classification happens synchronously BEFORE
    # the stream starts — so HTTP errors are returned normally, not buried in SSE.
    with _SL() as db:
        snapshot   = _load_pipeline_snapshot(data.user_id, db)

    classification = _classify_intent(data.message, data.history, snapshot)
    intent  = classification.get("intent", "general_startup_chat")
    section = classification.get("section")

    # Non-conversational intents: return a short non-streamed message
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
            yield f"data: {_json.dumps({'type': 'token', 'content': reply_text})}\n\n"
            yield f"data: {_json.dumps({'type': 'done', 'intent': intent, 'action': intent, 'route_to_trigger': trigger, 'trigger_needed': trigger is not None})}\n\n"

        return StreamingResponse(_instant(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if intent == "out_of_scope":
        def _decline():
            yield f"data: {_json.dumps({'type': 'token', 'content': OUT_OF_SCOPE_RESPONSE})}\n\n"
            yield f"data: {_json.dumps({'type': 'done', 'intent': intent, 'action': 'declined', 'route_to_trigger': None, 'trigger_needed': False})}\n\n"

        return StreamingResponse(_decline(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # Conversational intents — stream the LLM reply
    from System_Messages.general_bot_prompt import GENERAL_BOT_SYSTEM_PROMPT as _SYS

    with _SL() as db:
        snapshot     = _load_pipeline_snapshot(data.user_id, db)
        section_data = _load_section_data(data.user_id, section, db) if section else None

    from agents.generalBot import _snapshot_to_context_string

    context_parts = [_snapshot_to_context_string(snapshot)]
    if section_data and section:
        context_parts.append(
            f"\n=== DETAILED DATA FOR [{section}] ===\n"
            + _json.dumps(section_data, indent=2)[:4000]
        )

    messages = [
        {"role": "system", "content": _SYS},
        {"role": "system", "content": "\n\n".join(context_parts)},
        *data.history[-20:],
        {"role": "user",   "content": data.message},
    ]

    def _on_complete(full_reply: str) -> Dict[str, Any]:
        return {"intent": intent, "action": "answered", "route_to_trigger": None, "trigger_needed": False}

    return _sse_chat_stream(
        messages=messages,
        groq_client=groq_client,
        groq_model=GROQ_MODEL,
        temperature=0.4,
        max_tokens=600,
        on_complete=_on_complete,
    )


# ── Explainer Bot ─────────────────────────────────────────────────────────────

@router.post("/explain", dependencies=[Depends(verify_api_key)])
def explain(data: ExplainerInput) -> Dict[str, Any]:
    """
    Read-only explainer bot — helps the founder understand existing pipeline data.

    Unlike /general-chat, this bot:
    - Never routes to agents or triggers new generation
    - Only answers questions about already-generated data
    - Optionally loads a specific section in full for detailed explanation

    Use this for a "explain this section to me" sidebar in the UI.

    Request:
      user_id  : str
      message  : str   — the question (e.g. "what does my LTV/CAC ratio mean?")
      history  : list  — prior turns for context (optional)
      section  : str   — which section to load in full (optional)
                         e.g. "customers", "business_model", "unit_economics"
    """
    from agents.ExplainerBot import run_explainer_bot

    result = run_explainer_bot(
        user_id=data.user_id,
        message=data.message,
        history=data.history,
        section=data.section,
    )

    return {
        "user_id": data.user_id,
        **result,
        "chat_history_length": len(data.history) + 2,
    }
