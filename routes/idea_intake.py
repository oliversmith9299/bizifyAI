from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse

from db.connection import get_db
from db import crud
from routes.dependencies import verify_api_key, IdeaIntakeInput, UserIdInput

router = APIRouter()


@router.post("/idea-intake", dependencies=[Depends(verify_api_key)])
def idea_intake(data: IdeaIntakeInput):
    from agents.ThreeIdeaIntakeAgent import run_idea_intake

    result = run_idea_intake(
        user_id=data.user_id,
        user_message=data.message,
        history=data.history,
    )
    return {"user_id": data.user_id, **result}


@router.post("/idea-intake/run-problems/{user_id}", dependencies=[Depends(verify_api_key)])
def idea_intake_run_problems(user_id: str, background_tasks: BackgroundTasks, db=Depends(get_db)):
    intake = crud.get_idea_intake_json(db, user_id)
    if not intake or intake.get("_status") == "pending_clarification":
        raise HTTPException(status_code=425, detail="Idea intake not ready. Complete /idea-intake first.")

    crud.upsert_pipeline_status(db, user_id, "pending")

    from orchestrator.orchestrator import run_returning_user_pipeline
    background_tasks.add_task(run_returning_user_pipeline, user_id, intake)

    return JSONResponse(status_code=202, content={
        "user_id": user_id,
        "status":  "pending",
        "message": "Problem discovery started. Poll /pipeline/status/{user_id} for progress.",
        "poll_url": f"/pipeline/status/{user_id}",
    })


@router.post("/idea-intake/start-chat", dependencies=[Depends(verify_api_key)])
def idea_intake_start_chat(data: UserIdInput, db=Depends(get_db)):
    intake       = crud.get_idea_intake_json(db, data.user_id)
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

    profile_compat       = _build_profile_for_problem_discovery(intake)
    questionnaire_compat = _build_questionnaire_for_problem_discovery(intake)
    questionnaire_compat["skills"] = []

    crud.save_profile(db, data.user_id, profile_compat)
    crud.save_questionnaire_output(db, data.user_id, questionnaire_compat)

    context = build_context(problems_row.data, questionnaire_compat, [])
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
        {"role": "user",   "content": opening},
    ]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.4, max_tokens=1000,
    )
    reply = response.choices[0].message.content.strip()

    history = [
        {"role": "user",      "content": opening},
        {"role": "assistant", "content": reply},
    ]
    crud.save_idea(db, data.user_id, reply, history)
    crud.upsert_pipeline_status(db, data.user_id, "done", None)

    return {
        "user_id":      data.user_id,
        "status":       "done",
        "current_idea": reply,
        "chat_history": history,
        "message":      "Idea chat is ready. Continue with /pipeline/chat.",
    }


@router.get("/idea-intake/{user_id}", dependencies=[Depends(verify_api_key)])
def get_idea_intake(user_id: str, db=Depends(get_db)):
    intake = crud.get_idea_intake_json(db, user_id)
    if not intake:
        raise HTTPException(status_code=404, detail="No idea intake found for this user.")

    return {"user_id": user_id, "intake": intake}
