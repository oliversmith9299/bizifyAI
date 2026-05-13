from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/")
def health():
    return {"status": "AI service running"}
