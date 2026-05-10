"""
services/pipeline_service.py
=============================
Handles everything that belongs to the new-user pipeline flow.
Routes call this service instead of talking to the orchestrator directly.

DB integration notes (mapped to the platform schema PDF):
  - Saves questionnaire output   → questionnaire_outputs table (AI-private)
  - Saves profile analysis       → profile_results (AI-private) + agent_runs (shared)
  - Saves problems               → problems_results (AI-private) + agent_runs (shared)
  - Saves idea                   → idea_results (AI-private) + ideas table (platform)
                                   + agent_runs (shared)
  - Chat history                 → chat_sessions + chat_messages (platform schema)
  - Pipeline progress            → pipeline_runs (AI-private, mirrors roadmap_stages concept)
"""

import uuid
from datetime import datetime

from db.connection import SessionLocal
from db import crud
from services.base_service import BaseService


class PipelineService(BaseService):
    service_name = "pipeline_service"

    # ── New-user flow ─────────────────────────────────────────────────────────

    def handle_start(self, user_id: str, questionnaire: dict, skills: list) -> dict:
        """
        Save the questionnaire, mark pipeline pending, return 202 payload.
        The orchestrator will be scheduled as a BackgroundTask by the route.
        """
        db = SessionLocal()
        try:
            crud.save_questionnaire_output(db, user_id, questionnaire)
            crud.upsert_pipeline_status(db, user_id, "pending")
        finally:
            db.close()

        return {
            "user_id": user_id,
            "status": "pending",
            "message": "Pipeline started. Poll /pipeline/status/{user_id} for progress.",
            "poll_url": f"/pipeline/status/{user_id}",
        }

    def handle_status(self, user_id: str) -> dict:
        """Return full pipeline readiness for a user."""
        db = SessionLocal()
        try:
            run = crud.get_pipeline_status(db, user_id)
            if not run:
                return {"error": "not_found"}
            return {
                "user_id":       user_id,
                "status":        run.status,
                "current_step":  run.current_step,
                "profile_ready": crud.get_profile(db, user_id) is not None,
                "problems_ready":crud.get_problems(db, user_id) is not None,
                "intake_ready":  crud.get_idea_intake_json(db, user_id) is not None,
                "idea_ready":    crud.get_idea(db, user_id) is not None,
                "error":         run.error,
            }
        finally:
            db.close()

    # ── Chat ─────────────────────────────────────────────────────────────────

    def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Persist a single chat turn to the platform chat_messages table.
        Creates the session row if it doesn't exist yet.
        """
        from db.models import ChatSession, ChatMessage
        db = SessionLocal()
        try:
            session = db.query(ChatSession).filter_by(id=session_id).first()
            if not session:
                session = ChatSession(
                    id=session_id,
                    user_id=user_id,
                    session_type="idea_chat",
                    created_at=datetime.utcnow(),
                )
                db.add(session)
                db.flush()

            msg = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=role,
                content=content,
                created_at=datetime.utcnow(),
            )
            db.add(msg)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def handle(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Use handle_start / handle_status / save_chat_message")


class IdeaIntakeService(BaseService):
    service_name = "idea_intake_service"

    def handle(self, user_id: str, message: str, history: list) -> dict:
        """Run IdeaIntake and return the structured result."""
        from agents.ThreeIdeaIntakeAgent import run_idea_intake
        return run_idea_intake(user_id, message, history)
