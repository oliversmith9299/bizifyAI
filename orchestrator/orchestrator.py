"""
orchestrator/orchestrator.py
============================
The SINGLE entry point that runs the pipeline for any user.

─── Role split ───────────────────────────────────────────────────────────────
  agents/PipelineRunner.py   → "toolbox"  — individual agent functions
                                (run_profile_analysis, run_problem_discovery, …)
  orchestrator/session.py    → "state"    — tracks which step is current,
                                which flow (new vs returning), advance/fail
  orchestrator/orchestrator.py → "manager" — decides what to call and when,
                                saves every result to DB, handles errors

─── Flows ────────────────────────────────────────────────────────────────────
  New user      : ProfileAnalysis → ProblemDiscovery → IdeaChat → steps 4-12
  Returning user: IdeaIntake (already done by route) →
                  ProblemDiscovery → IdeaChat → steps 4-12

─── Called by ────────────────────────────────────────────────────────────────
  routes/main.py  /pipeline/run                   → run_new_user_pipeline()
  routes/main.py  /idea-intake/run-problems/{uid} → run_returning_user_pipeline()
"""

import logging

from db.connection import get_session
from db import crud
from orchestrator.session import Flow, make_session

log = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# New-user flow  (questionnaire → profile → problems → idea)
# ─────────────────────────────────────────────────────────────────────────────
async def run_new_user_pipeline(user_id: str, questionnaire: dict, skills: list):
    """
    Full pipeline for a first-time user.
    Triggered as a FastAPI BackgroundTask by POST /pipeline/run.
    """
    from agents.PipelineRunner import (
        run_profile_analysis,
        run_problem_discovery,
        generate_opening_idea,
        build_context,
    )

    session = make_session(user_id, has_idea=False)

    try:
        # ── Step 1 · Profile Analysis ────────────────────────────────────────
        _status(db=None, user_id=user_id, step=session)
        log.info(f"[{user_id}] ▶ {session.step_label}")

        profile = run_profile_analysis(questionnaire, skills)

        with get_session() as db:
            crud.save_profile(db, user_id, profile)
            crud.upsert_pipeline_status(db, user_id, "running", session.current_step.value)

        log.info(f"[{user_id}] ✅ {session.step_label} done")
        session.advance(profile)

        # ── Step 2 · Problem Discovery ───────────────────────────────────────
        log.info(f"[{user_id}] ▶ {session.step_label}")

        problems = run_problem_discovery(profile, questionnaire)

        with get_session() as db:
            crud.save_problems(db, user_id, problems)
            crud.upsert_pipeline_status(db, user_id, "running", session.current_step.value)

        log.info(f"[{user_id}] ✅ {session.step_label} done — "
                 f"{len(problems.get('problems', []))} problems found")
        session.advance(problems)

        # ── Step 3 · Idea Generation ─────────────────────────────────────────
        log.info(f"[{user_id}] ▶ {session.step_label}")

        context = build_context(problems, questionnaire, skills)
        idea    = generate_opening_idea(context)

        with get_session() as db:
            crud.save_idea(db, user_id, idea, [])
            crud.upsert_pipeline_status(db, user_id, "done", None)

        session.advance(idea)
        log.info(f"[{user_id}] ✅ Pipeline complete — idea saved")

    except Exception as e:
        session.fail(str(e))
        log.error(f"[{user_id}] ❌ Pipeline failed at '{session.step_label}': {e}",
                  exc_info=True)
        _save_error(user_id, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Returning-user flow  (intake already saved → problems → idea)
# ─────────────────────────────────────────────────────────────────────────────
async def run_returning_user_pipeline(user_id: str, intake_data: dict):
    """
    Pipeline for a user who already has an idea.
    IdeaIntake was already run by the /idea-intake route.
    This picks up at ProblemDiscovery and finishes at IdeaChat.
    Triggered as a FastAPI BackgroundTask by POST /idea-intake/run-problems/{user_id}.
    """
    from agents.PipelineRunner import run_problem_discovery
    from agents.ThreeIdeaIntakeAgent import (
        _build_profile_for_problem_discovery,
        _build_questionnaire_for_problem_discovery,
    )

    session = make_session(user_id, has_idea=True)
    session.advance()   # IDEA_INTAKE is already done — skip to PROBLEM_DISCOVERY

    profile_compat       = _build_profile_for_problem_discovery(intake_data)
    questionnaire_compat = _build_questionnaire_for_problem_discovery(intake_data)

    try:
        # ── Step · Problem Discovery ─────────────────────────────────────────
        log.info(f"[{user_id}] ▶ {session.step_label}")

        with get_session() as db:
            crud.upsert_pipeline_status(db, user_id, "running", session.current_step.value)

        problems = run_problem_discovery(profile_compat, questionnaire_compat)

        with get_session() as db:
            crud.save_problems(db, user_id, problems)
            crud.upsert_pipeline_status(db, user_id, "problems_done", "idea_chat_start")

        session.advance(problems)
        log.info(f"[{user_id}] ✅ {session.step_label} done — "
                 f"{len(problems.get('problems', []))} problems found")

    except Exception as e:
        session.fail(str(e))
        log.error(f"[{user_id}] ❌ Pipeline failed at '{session.step_label}': {e}",
                  exc_info=True)
        _save_error(user_id, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save_error(user_id: str, message: str):
    try:
        with get_session() as db:
            crud.upsert_pipeline_status(db, user_id, "error", None, message)
    except Exception as db_err:
        log.error(f"[{user_id}] Also failed to save error status: {db_err}")


def _status(db, user_id: str, step):
    """Fire-and-forget status update before a step starts."""
    try:
        with get_session() as s:
            crud.upsert_pipeline_status(s, user_id, "running", step.current_step.value)
    except Exception:
        pass
