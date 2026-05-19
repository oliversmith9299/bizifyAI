from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from db.connection import get_db, SessionLocal
from db import crud
from routes.dependencies import (
    verify_api_key,
    sse_chat_stream,
    QuestionnaireInput,
    ChatInput,
)

router = APIRouter()


@router.post("/run", dependencies=[Depends(verify_api_key)])
async def run_pipeline(
    data: QuestionnaireInput,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
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
        "user_id":   data.user_id,
        "status":    "pending",
        "message":   "Pipeline started. Poll /pipeline/status/{user_id} for progress.",
        "poll_url":  f"/pipeline/status/{data.user_id}",
    })


@router.get("/status/{user_id}", dependencies=[Depends(verify_api_key)])
def get_status(user_id: str, db=Depends(get_db)):
    run = crud.get_pipeline_status(db, user_id)
    if not run:
        raise HTTPException(status_code=404, detail="No pipeline found for this user_id")

    profile_ready          = crud.get_profile(db, user_id) is not None
    problems_ready         = crud.get_problems(db, user_id) is not None
    intake_ready           = crud.get_idea_intake_json(db, user_id) is not None
    idea_ready             = crud.get_idea(db, user_id) is not None
    customers_ready        = crud.get_customers_json(db, user_id) is not None
    competition_ready      = crud.get_competition_json(db, user_id) is not None
    market_potential_ready = crud.get_market_potential_json(db, user_id) is not None
    idea_strategy_ready    = crud.get_idea_strategy_json(db, user_id) is not None
    business_model_ready   = crud.get_business_model_json(db, user_id) is not None
    functions_list_ready   = crud.get_functions_list_json(db, user_id) is not None
    mvp_planning_ready     = crud.get_mvp_planning_json(db, user_id) is not None
    unit_economics_ready   = crud.get_unit_economics_json(db, user_id) is not None
    go_to_market_ready     = crud.get_go_to_market_json(db, user_id) is not None

    pipeline_complete = all([
        profile_ready, problems_ready, idea_ready,
        customers_ready, competition_ready, market_potential_ready,
        idea_strategy_ready, business_model_ready, functions_list_ready,
        mvp_planning_ready, unit_economics_ready, go_to_market_ready,
    ])

    return {
        "user_id":                 user_id,
        "status":                  run.status,
        "current_step":            run.current_step,
        "profile_ready":           profile_ready,
        "problems_ready":          problems_ready,
        "intake_ready":            intake_ready,
        "idea_ready":              idea_ready,
        "customers_ready":         customers_ready,
        "competition_ready":       competition_ready,
        "market_potential_ready":  market_potential_ready,
        "idea_strategy_ready":     idea_strategy_ready,
        "business_model_ready":    business_model_ready,
        "functions_list_ready":    functions_list_ready,
        "mvp_planning_ready":      mvp_planning_ready,
        "unit_economics_ready":    unit_economics_ready,
        "go_to_market_ready":      go_to_market_ready,
        "pipeline_complete":       pipeline_complete,
        "error":                   run.error,
    }


@router.get("/idea/{user_id}", dependencies=[Depends(verify_api_key)])
def get_idea(user_id: str, db=Depends(get_db)):
    idea = crud.get_idea(db, user_id)
    if not idea:
        raise HTTPException(status_code=425, detail="Idea not ready yet — pipeline still running")

    return {
        "user_id":      user_id,
        "current_idea": idea.current_idea,
        "chat_history": idea.chat_history or [],
    }


@router.post("/chat", dependencies=[Depends(verify_api_key)])
def chat(data: ChatInput, db=Depends(get_db)):
    run = crud.get_pipeline_status(db, data.user_id)
    if not run or run.status != "done":
        raise HTTPException(status_code=425, detail="Pipeline not ready yet. Call /pipeline/run first.")

    idea_row          = crud.get_idea(db, data.user_id)
    profile_row       = crud.get_profile(db, data.user_id)
    problems_row      = crud.get_problems(db, data.user_id)
    questionnaire_row = crud.get_questionnaire_output(db, data.user_id)

    if not (idea_row and profile_row and problems_row and questionnaire_row):
        raise HTTPException(status_code=425, detail="Missing required context in DB to chat.")

    from agents.PipelineRunner import build_context
    stored_questionnaire = questionnaire_row.data
    context = build_context(problems_row.data, stored_questionnaire, stored_questionnaire.get("skills", []))

    intake = crud.get_idea_intake_json(db, data.user_id)
    if intake:
        context += f"\n\n=== USER ORIGINAL IDEA INTAKE ===\n{intake}"

    from agents.PipelineRunner import groq_client, GROQ_MODEL, IDEA_SYSTEM_PROMPT

    existing_history = idea_row.chat_history or []
    messages = [
        {"role": "system", "content": IDEA_SYSTEM_PROMPT},
        {"role": "system", "content": context},
        *existing_history[-20:],
        {"role": "user", "content": data.message},
    ]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.4, max_tokens=1000,
    )
    reply = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens if response.usage else 0

    new_idea = reply if "💡 IDEA:" in reply else idea_row.current_idea
    existing_history.append({"role": "user",      "content": data.message})
    existing_history.append({"role": "assistant",  "content": reply})
    crud.save_idea(db, data.user_id, new_idea, existing_history)

    return {
        "user_id":             data.user_id,
        "reply":               reply,
        "chat_history_length": len(existing_history),
        "tokens_used":         tokens_used,
    }


@router.post("/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_stream(data: ChatInput, db=Depends(get_db)) -> StreamingResponse:
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
    context = build_context(problems_row.data, stored_questionnaire, stored_questionnaire.get("skills", []))

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

    user_id          = data.user_id
    current_idea_txt = idea_row.current_idea

    def _on_complete(full_reply: str) -> Dict[str, Any]:
        new_idea = full_reply if "💡 IDEA:" in full_reply else current_idea_txt
        updated  = existing_history + [
            {"role": "user",      "content": data.message},
            {"role": "assistant", "content": full_reply},
        ]
        with SessionLocal() as s:
            crud.save_idea(s, user_id, new_idea, updated)
        return {"chat_history_length": len(updated)}

    return sse_chat_stream(
        messages=messages, groq_client=groq_client, groq_model=GROQ_MODEL,
        temperature=0.4, max_tokens=1000, on_complete=_on_complete,
    )


@router.get("/questionnaire/{user_id}", dependencies=[Depends(verify_api_key)])
def get_questionnaire(user_id: str, db=Depends(get_db)):
    row = crud.get_questionnaire_output(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No questionnaire found")

    return {"user_id": user_id, "questionnaire": row.data}


@router.get("/profile/{user_id}", dependencies=[Depends(verify_api_key)])
def get_profile(user_id: str, db=Depends(get_db)):
    profile = crud.get_profile(db, user_id)
    if not profile:
        raise HTTPException(status_code=425, detail="Profile not ready yet")

    return {"user_id": user_id, "profile_analysis": profile.data}


@router.get("/problems/{user_id}", dependencies=[Depends(verify_api_key)])
def get_problems(user_id: str, db=Depends(get_db)):
    problems = crud.get_problems(db, user_id)
    if not problems:
        raise HTTPException(status_code=425, detail="Problems not ready yet")

    return {"user_id": user_id, "problems": problems.data}


@router.post("/rerun/profile/{user_id}", dependencies=[Depends(verify_api_key)])
def rerun_profile(user_id: str, db=Depends(get_db)):
    from agents.PipelineRunner import run_profile_analysis

    questionnaire_row = crud.get_questionnaire_output(db, user_id)
    if not questionnaire_row:
        raise HTTPException(404, "No questionnaire")

    data   = questionnaire_row.data
    skills = data.get("skills", [])
    profile = run_profile_analysis(data, skills)
    crud.save_profile(db, user_id, profile)

    return {"status": "profile regenerated"}


@router.post("/rerun/problems/{user_id}", dependencies=[Depends(verify_api_key)])
def rerun_problems(user_id: str, db=Depends(get_db)):
    from agents.PipelineRunner import run_problem_discovery

    profile       = crud.get_profile(db, user_id)
    questionnaire = crud.get_questionnaire_output(db, user_id)

    if not profile or not questionnaire:
        raise HTTPException(400, "Missing data")

    problems = run_problem_discovery(profile.data, questionnaire.data)
    crud.save_problems(db, user_id, problems)

    return {"status": "problems regenerated"}
