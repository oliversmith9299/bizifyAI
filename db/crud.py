"""
db/crud.py
==========
All database read/write operations for the AI service.

Each function opens no session — callers pass the session in so they control
the transaction boundary (route handlers, orchestrator steps, background tasks).
"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from db.models import (
    Agent,
    AgentRun,
    CompetitionResult,
    CustomersResult,
    IdeaIntakeResult,
    IdeaResult,
    IdeaStrategyResult,
    MarketPotentialResult,
    PipelineRun,
    ProfileResult,
    ProblemsResult,
    QuestionnaireOutput,
)


# Registry of all agents the AI service uses.
# Keys must match the names stored in the `agents` DB table.
AGENT_DEFINITIONS: dict[str, str] = {
    "OneProfileAnalysis":      "discovery",
    "TwoProblemDiscovery":     "discovery",
    "ThreeIdeaIntakeAgent":    "discovery",
    "ThreePersonalizeIdeaChat": "ideation",
    "FourCustomersAgent":      "planning",
    "FiveCompetitionAgent":    "planning",
    "SixMaketPotential":       "planning",
    "SevenIdeaStrategy":       "strategy",
    "EightBusinessModel":      "business",
    "NineFunctionsList":       "product",
    "TenMVPPlanning":          "product",
    "ElevenUnitEconomicsAgent": "finance",
    "TwelveGoToMarket":        "launch",
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_commit(db: Session, row: Any) -> Any:
    """Commit the session, refresh `row`, and roll back on any failure."""
    try:
        db.commit()
        db.refresh(row)
        return row
    except Exception:
        db.rollback()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline status
# ─────────────────────────────────────────────────────────────────────────────

def upsert_pipeline_status(
    db: Session,
    user_id: str,
    status: str,
    step: Optional[str] = None,
    error: Optional[str] = None,
) -> PipelineRun:
    """Create or update the pipeline status row for a user."""
    run = db.query(PipelineRun).filter_by(user_id=user_id).first()
    if not run:
        run = PipelineRun(user_id=user_id)
        db.add(run)

    run.status       = status
    run.current_step = step
    run.error        = error
    run.updated_at   = datetime.utcnow()

    return _safe_commit(db, run)


def get_pipeline_status(db: Session, user_id: str) -> Optional[PipelineRun]:
    """Return the pipeline status row for a user, or None."""
    return db.query(PipelineRun).filter_by(user_id=user_id).first()


# ─────────────────────────────────────────────────────────────────────────────
# Agent registry & runs
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_agent(
    db: Session,
    name: str,
    phase: Optional[str] = None,
) -> Agent:
    """Return the Agent row for `name`, creating it if it does not exist."""
    row = db.query(Agent).filter_by(name=name).first()
    if not row:
        row = Agent(name=name, phase=phase or AGENT_DEFINITIONS.get(name))
        db.add(row)
        db.flush()
    elif phase and row.phase != phase:
        row.phase = phase
    return row


def seed_agents(db: Session) -> None:
    """Ensure every agent in AGENT_DEFINITIONS has a row in the `agents` table."""
    for name, phase in AGENT_DEFINITIONS.items():
        get_or_create_agent(db, name, phase)
    db.commit()


def save_agent_run(
    db: Session,
    user_id: str,
    agent_name: str,
    input_data: Optional[dict] = None,
    output_data: Optional[dict] = None,
    status: str = "done",
    stage_id: Optional[str] = None,
    confidence_score: Optional[float] = None,
    execution_time_ms: Optional[int] = None,
) -> AgentRun:
    """
    Persist one agent execution to the `agent_runs` table.

    `user_id` is embedded inside input_data/output_data so
    get_latest_agent_run() can filter by user without a direct FK.
    """
    agent = get_or_create_agent(db, agent_name, AGENT_DEFINITIONS.get(agent_name))

    row = AgentRun(
        stage_id          = stage_id,
        agent_id          = agent.id,
        status            = status,
        input_data        = {"user_id": user_id, "payload": input_data or {}},
        output_data       = {"user_id": user_id, "payload": output_data or {}},
        confidence_score  = confidence_score,
        execution_time_ms = execution_time_ms,
    )
    db.add(row)
    return _safe_commit(db, row)


def get_latest_agent_run(
    db: Session,
    user_id: str,
    agent_name: str,
) -> Optional[AgentRun]:
    """Return the most recent AgentRun for this user and agent, or None."""
    agent = db.query(Agent).filter_by(name=agent_name).first()
    if not agent:
        return None

    rows = (
        db.query(AgentRun)
        .filter_by(agent_id=agent.id)
        .order_by(AgentRun.created_at.desc())
        .limit(100)
        .all()
    )
    for row in rows:
        if (row.input_data or {}).get("user_id") == user_id:
            return row
        if (row.output_data or {}).get("user_id") == user_id:
            return row
    return None


def get_latest_agent_output(
    db: Session,
    user_id: str,
    agent_name: str,
) -> Optional[dict]:
    """Return the output payload dict from the latest run, or None."""
    row = get_latest_agent_run(db, user_id, agent_name)
    if not row:
        return None
    return (row.output_data or {}).get("payload")


# ─────────────────────────────────────────────────────────────────────────────
# Profile
# ─────────────────────────────────────────────────────────────────────────────

def save_profile(db: Session, user_id: str, data: dict) -> ProfileResult:
    """Persist profile analysis output and record an agent_run entry."""
    save_agent_run(db, user_id=user_id, agent_name="OneProfileAnalysis", output_data=data)

    row = db.query(ProfileResult).filter_by(user_id=user_id).first()
    if not row:
        row = ProfileResult(user_id=user_id)
        db.add(row)

    row.data = data
    return _safe_commit(db, row)


def get_profile(db: Session, user_id: str) -> Optional[ProfileResult]:
    return db.query(ProfileResult).filter_by(user_id=user_id).first()


# ─────────────────────────────────────────────────────────────────────────────
# Questionnaire
# ─────────────────────────────────────────────────────────────────────────────

def save_questionnaire_output(db: Session, user_id: str, data: dict) -> QuestionnaireOutput:
    """Persist the questionnaire payload sent by the backend."""
    row = db.query(QuestionnaireOutput).filter_by(user_id=user_id).first()
    if not row:
        row = QuestionnaireOutput(user_id=user_id)
        db.add(row)

    row.data       = data
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_questionnaire_output(db: Session, user_id: str) -> Optional[QuestionnaireOutput]:
    return db.query(QuestionnaireOutput).filter_by(user_id=user_id).first()


def get_questionnaire_output_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_questionnaire_output(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Problems
# ─────────────────────────────────────────────────────────────────────────────

def save_problems(db: Session, user_id: str, data: dict) -> ProblemsResult:
    """Persist problem discovery output and record an agent_run entry."""
    save_agent_run(db, user_id=user_id, agent_name="TwoProblemDiscovery", output_data=data)

    row = db.query(ProblemsResult).filter_by(user_id=user_id).first()
    if not row:
        row = ProblemsResult(user_id=user_id)
        db.add(row)

    row.data = data
    return _safe_commit(db, row)


def get_problems(db: Session, user_id: str) -> Optional[ProblemsResult]:
    return db.query(ProblemsResult).filter_by(user_id=user_id).first()


# ─────────────────────────────────────────────────────────────────────────────
# Idea intake  (returning-user flow)
# ─────────────────────────────────────────────────────────────────────────────

def save_idea_intake(db: Session, user_id: str, data: dict) -> IdeaIntakeResult:
    """Persist idea intake output. Status is 'pending_clarification' if still in chat."""
    is_pending = data.get("_status") == "pending_clarification"
    save_agent_run(
        db,
        user_id=user_id,
        agent_name="ThreeIdeaIntakeAgent",
        output_data=data,
        status="pending_clarification" if is_pending else "done",
    )

    row = db.query(IdeaIntakeResult).filter_by(user_id=user_id).first()
    if not row:
        row = IdeaIntakeResult(user_id=user_id)
        db.add(row)

    row.data       = data
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_idea_intake(db: Session, user_id: str) -> Optional[IdeaIntakeResult]:
    return db.query(IdeaIntakeResult).filter_by(user_id=user_id).first()


def get_idea_intake_json(db: Session, user_id: str) -> Optional[dict]:
    """Return the intake payload, preferring the agent_run record for consistency."""
    agent_output = get_latest_agent_output(db, user_id, "ThreeIdeaIntakeAgent")
    if agent_output:
        return agent_output
    row = get_idea_intake(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Idea
# ─────────────────────────────────────────────────────────────────────────────

def save_idea(db: Session, user_id: str, idea: str, history: list) -> IdeaResult:
    """Persist the current idea text and full chat history."""
    save_agent_run(
        db,
        user_id=user_id,
        agent_name="ThreePersonalizeIdeaChat",
        output_data={"current_idea": idea, "chat_history": history},
    )

    row = db.query(IdeaResult).filter_by(user_id=user_id).first()
    if not row:
        row = IdeaResult(user_id=user_id)
        db.add(row)

    row.current_idea = idea
    row.chat_history = history
    row.updated_at   = datetime.utcnow()
    return _safe_commit(db, row)


def get_idea(db: Session, user_id: str) -> Optional[IdeaResult]:
    return db.query(IdeaResult).filter_by(user_id=user_id).first()


# ─────────────────────────────────────────────────────────────────────────────
# Section results  (agents 4-12, same pattern per section)
# ─────────────────────────────────────────────────────────────────────────────

def _save_section(db: Session, user_id: str, row_cls: Any, data: dict, chat_history: Optional[list]) -> Any:
    """Generic upsert for any section result table."""
    row = db.query(row_cls).filter_by(user_id=user_id).first()
    if not row:
        row = row_cls(user_id=user_id)
        db.add(row)

    row.data         = data
    row.chat_history = chat_history if chat_history is not None else (row.chat_history or [])
    row.updated_at   = datetime.utcnow()
    return _safe_commit(db, row)


# Customers ───────────────────────────────────────────────────────────────────

def save_customers(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> CustomersResult:
    return _save_section(db, user_id, CustomersResult, data, chat_history)


def get_customers(db: Session, user_id: str) -> Optional[CustomersResult]:
    return db.query(CustomersResult).filter_by(user_id=user_id).first()


def get_customers_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_customers(db, user_id)
    return row.data if row else None


# Competition ─────────────────────────────────────────────────────────────────

def save_competition(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> CompetitionResult:
    return _save_section(db, user_id, CompetitionResult, data, chat_history)


def get_competition(db: Session, user_id: str) -> Optional[CompetitionResult]:
    return db.query(CompetitionResult).filter_by(user_id=user_id).first()


def get_competition_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_competition(db, user_id)
    return row.data if row else None


# Market potential ────────────────────────────────────────────────────────────

def save_market_potential(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> MarketPotentialResult:
    return _save_section(db, user_id, MarketPotentialResult, data, chat_history)


def get_market_potential(db: Session, user_id: str) -> Optional[MarketPotentialResult]:
    return db.query(MarketPotentialResult).filter_by(user_id=user_id).first()


def get_market_potential_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_market_potential(db, user_id)
    return row.data if row else None


# Idea strategy ───────────────────────────────────────────────────────────────

def save_idea_strategy(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> IdeaStrategyResult:
    return _save_section(db, user_id, IdeaStrategyResult, data, chat_history)


def get_idea_strategy(db: Session, user_id: str) -> Optional[IdeaStrategyResult]:
    return db.query(IdeaStrategyResult).filter_by(user_id=user_id).first()


def get_idea_strategy_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_idea_strategy(db, user_id)
    return row.data if row else None
