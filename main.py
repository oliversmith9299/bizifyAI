from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.config import ALLOWED_ORIGINS
from db.connection import engine
from db import crud
from db.models import (
    UserProfile,          # backend-owned — imported so the ORM mapper registers it
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
    BusinessModelResult,
    FunctionsListResult,
    MVPPlanningResult,
    UnitEconomicsResult,
    GoToMarketResult,
)
from sqlalchemy import MetaData, Table
from routes import router as pipeline_router

app = FastAPI(
    title="Bizify AI Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — the AI service must only be called by the backend, never by browsers.
# Set ALLOWED_ORIGINS=https://your-backend-host in .env for production.
# When ALLOWED_ORIGINS is empty (default), no browser origin is allowed.
_cors_origins = ALLOWED_ORIGINS if ALLOWED_ORIGINS else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["x-api-key", "Content-Type"],
)

app.include_router(pipeline_router)


# ── AI-private table registry ──────────────────────────────────────────────
# Tables the AI service owns and is allowed to auto-create.
# Platform tables (businesses, ideas, chat_sessions, etc.) are owned by the
# backend — the AI service must NOT create or alter them.
_AI_OWNED_TABLES = {
    # Pipeline orchestration
    "pipeline_runs",
    # Founder profiling
    "profile_results",
    "questionnaire_outputs",
    "problems_results",
    # Idea discovery
    "idea_results",
    "idea_intake_results",
    # Analysis sections (agents 4-12)
    "customers_results",
    "competition_results",
    "market_potential_results",
    "idea_strategy_results",
    "business_model_results",
    "functions_list_results",
    "mvp_planning_results",
    "unit_economics_results",
    "go_to_market_results",
    # user_profiles is backend-owned — NOT listed here so create_all never touches it
}


@app.on_event("startup")
def ensure_ai_tables():
    """
    Create only the AI-service-owned tables.
    Platform tables (businesses, ideas, roadmap_stages, chat_sessions, etc.)
    are managed by the backend migrations — never touch them here.
    """
    from sqlalchemy import inspect, text
    from db.connection import Base

    inspector   = inspect(engine)
    existing    = set(inspector.get_table_names())
    ai_metadata = MetaData()

    # Copy only AI-owned table definitions into a separate metadata object
    for table in Base.metadata.sorted_tables:
        if table.name in _AI_OWNED_TABLES and table.name not in existing:
            table.tometadata(ai_metadata)

    ai_metadata.create_all(bind=engine)

    # Add sources_list column to existing tables that don't have it yet
    _tables_needing_sources_list = [
        "problems_results",
        "customers_results",
        "competition_results",
        "market_potential_results",
        "idea_strategy_results",
        "business_model_results",
        "functions_list_results",
        "mvp_planning_results",
        "unit_economics_results",
        "go_to_market_results",
    ]
    with engine.connect() as conn:
        for tbl in _tables_needing_sources_list:
            if tbl not in existing:
                continue
            cols = {c["name"] for c in inspector.get_columns(tbl)}
            if "sources_list" not in cols:
                conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN sources_list JSON"))
        conn.commit()


@app.get("/")
def health():
    return {"status": "AI service running"}
