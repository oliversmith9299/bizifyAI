from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.connection import engine
from db import crud
from db.models import (
    # AI-private tables — safe to auto-create
    PipelineRun,
    ProfileResult,
    QuestionnaireOutput,
    ProblemsResult,
    IdeaResult,
    IdeaIntakeResult,
    CustomersResult,
    CompetitionResult,
    MarketPotentialResult,
    IdeaStrategyResult,
    # Shared platform tables — already managed by the backend migrations
    # Agent and AgentRun are in the DB schema PDF (Section 4) — backend owns them
    # We only auto-create them here if they don't already exist (safe: create_all is idempotent)
    Agent,
    AgentRun,
)
from sqlalchemy import MetaData, Table
from routes.main import router as pipeline_router

app = FastAPI(
    title="Bizify AI Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — only the backend server should call this service in production.
# In development, allow localhost ports for testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to backend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router)


# ── AI-private table registry ──────────────────────────────────────────────
# Tables the AI service owns and is allowed to auto-create.
# Platform tables (businesses, ideas, chat_sessions, etc.) are owned by the
# backend — the AI service must NOT create or alter them.
_AI_OWNED_TABLES = {
    "pipeline_runs",
    "profile_results",
    "questionnaire_outputs",
    "problems_results",
    "idea_results",
    "idea_intake_results",
    "customers_results",
    "competition_results",
    "market_potential_results",
    "idea_strategy_results",
    "agents",
    "agent_runs",
}


@app.on_event("startup")
def ensure_ai_tables():
    """
    Create only the AI-service-owned tables.
    Platform tables (businesses, ideas, roadmap_stages, chat_sessions, etc.)
    are managed by the backend migrations — never touch them here.
    """
    from sqlalchemy import inspect
    from db.connection import Base

    inspector   = inspect(engine)
    existing    = set(inspector.get_table_names())
    ai_metadata = MetaData()

    # Copy only AI-owned table definitions into a separate metadata object
    for table in Base.metadata.sorted_tables:
        if table.name in _AI_OWNED_TABLES and table.name not in existing:
            table.tometadata(ai_metadata)

    ai_metadata.create_all(bind=engine)

    # Seed agent registry rows
    from db.connection import SessionLocal
    session = SessionLocal()
    try:
        crud.seed_agents(session)
    finally:
        session.close()


@app.get("/")
def health():
    return {"status": "AI service running"}
