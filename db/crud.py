"""
db/crud.py
==========
All database read/write operations for the AI service.

Each agent has its own dedicated table with columns that match its JSON output.
Callers pass the session in so they control the transaction boundary.
"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from db.models import (
    UserProfile,
    BusinessModelResult,
    CompetitionResult,
    CustomersResult,
    FunctionsListResult,
    GoToMarketResult,
    IdeaIntakeResult,
    IdeaResult,
    IdeaStrategyResult,
    MarketPotentialResult,
    MVPPlanningResult,
    PipelineRun,
    ProfileResult,
    ProblemsResult,
    QuestionnaireOutput,
    UnitEconomicsResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_commit(db: Session, row: Any) -> Any:
    try:
        db.commit()
        db.refresh(row)
        return row
    except Exception:
        db.rollback()
        raise


def _upsert(db: Session, model_cls: Any, user_id: str) -> Any:
    """Fetch existing row or create a new one (not yet committed)."""
    row = db.query(model_cls).filter_by(user_id=user_id).first()
    if not row:
        row = model_cls(user_id=user_id)
        db.add(row)
    return row


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
    return db.query(PipelineRun).filter_by(user_id=user_id).first()


# ─────────────────────────────────────────────────────────────────────────────
# user_profiles  (backend-owned — AI reads only; never writes)
# ─────────────────────────────────────────────────────────────────────────────

def get_user_profile(db: Session, user_id: str) -> Optional[UserProfile]:
    """Return the user_profiles row for this user, or None."""
    return db.query(UserProfile).filter_by(user_id=user_id).first()


def get_questionnaire_from_profile(db: Session, user_id: str) -> Optional[dict]:
    """
    Extract the questionnaire payload stored in user_profiles.questionnaire_json.
    Returns None if the row does not exist or the column is empty.
    """
    row = get_user_profile(db, user_id)
    return row.questionnaire_json if row else None


def get_skills_from_profile(db: Session, user_id: str) -> list:
    """
    Extract the skills list stored in user_profiles.skills_json.
    Returns [] if not found.
    """
    row = get_user_profile(db, user_id)
    if not row:
        return []
    skills = row.skills_json
    if isinstance(skills, list):
        return skills
    if isinstance(skills, dict):
        return skills.get("skills", [])
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Questionnaire  (raw input — stored as a single blob, not an agent output)
# ─────────────────────────────────────────────────────────────────────────────

def save_questionnaire_output(db: Session, user_id: str, data: dict) -> QuestionnaireOutput:
    row = _upsert(db, QuestionnaireOutput, user_id)
    row.data       = data
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_questionnaire_output(db: Session, user_id: str) -> Optional[QuestionnaireOutput]:
    return db.query(QuestionnaireOutput).filter_by(user_id=user_id).first()


def get_questionnaire_output_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_questionnaire_output(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — OneProfileAnalysis → profile_results
# ─────────────────────────────────────────────────────────────────────────────

def save_profile(db: Session, user_id: str, data: dict) -> ProfileResult:
    row = _upsert(db, ProfileResult, user_id)
    row.personality_insights       = data.get("personality_insights")
    row.founder_profile            = data.get("founder_profile")
    row.recommended_industries     = data.get("recommended_industries")
    row.recommended_problem_spaces = data.get("recommended_problem_spaces")
    row.search_direction           = data.get("search_direction")
    row.system_flags               = data.get("system_flags")
    return _safe_commit(db, row)


def get_profile(db: Session, user_id: str) -> Optional[ProfileResult]:
    return db.query(ProfileResult).filter_by(user_id=user_id).first()


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — TwoProblemDiscovery → problems_results
# ─────────────────────────────────────────────────────────────────────────────

def save_problems(db: Session, user_id: str, data: dict) -> ProblemsResult:
    row = _upsert(db, ProblemsResult, user_id)
    row.problems          = data.get("problems")
    row.customer_segments = data.get("customer_segments")
    row.personas          = data.get("personas")
    row.summary_insight   = data.get("summary_insight")
    return _safe_commit(db, row)


def get_problems(db: Session, user_id: str) -> Optional[ProblemsResult]:
    return db.query(ProblemsResult).filter_by(user_id=user_id).first()


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3a — ThreePersonalizeIdeaChat → idea_results
# ─────────────────────────────────────────────────────────────────────────────

def save_idea(db: Session, user_id: str, idea: str, history: list) -> IdeaResult:
    row = _upsert(db, IdeaResult, user_id)
    row.current_idea = idea
    row.chat_history = history
    row.updated_at   = datetime.utcnow()
    return _safe_commit(db, row)


def get_idea(db: Session, user_id: str) -> Optional[IdeaResult]:
    return db.query(IdeaResult).filter_by(user_id=user_id).first()


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3b — ThreeIdeaIntakeAgent → idea_intake_results
# ─────────────────────────────────────────────────────────────────────────────

def save_idea_intake(db: Session, user_id: str, data: dict) -> IdeaIntakeResult:
    """
    `data` is the flat intake dict (the `intake` or `partial_intake` sub-object),
    plus an optional `_decision` key injected by the route.
    """
    intake   = data.get("intake") or data.get("partial_intake") or data
    decision = data.get("decision", "ready")

    row = _upsert(db, IdeaIntakeResult, user_id)
    row.decision                       = decision
    row.idea_summary                   = intake.get("idea_summary")
    row.target_users                   = intake.get("target_users")
    row.industry                       = intake.get("industry")
    row.problem_assumption             = intake.get("problem_assumption")
    row.solution_assumption            = intake.get("solution_assumption")
    row.business_model                 = intake.get("business_model")
    row.region                         = intake.get("region")
    row.keywords_for_problem_discovery = intake.get("keywords_for_problem_discovery")
    row.unclear_questions              = intake.get("unclear_questions")
    row.reply                          = data.get("reply")
    row.updated_at                     = datetime.utcnow()
    return _safe_commit(db, row)


def get_idea_intake(db: Session, user_id: str) -> Optional[IdeaIntakeResult]:
    return db.query(IdeaIntakeResult).filter_by(user_id=user_id).first()


def get_idea_intake_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_idea_intake(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — FourCustomersAgent → customers_results
# ─────────────────────────────────────────────────────────────────────────────

def save_customers(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> CustomersResult:
    row = _upsert(db, CustomersResult, user_id)
    row.customer_segments     = data.get("customer_segments")
    row.primary_segment       = data.get("primary_segment")
    row.catwoe                = data.get("catwoe")
    row.personas              = data.get("personas")
    row.acquisition_channels  = data.get("acquisition_channels")
    row.early_adopter_profile = data.get("early_adopter_profile")
    row.summary               = data.get("summary")
    row.source_mode           = data.get("source_mode")
    row.sources_used          = data.get("sources_used")
    if chat_history is not None:
        row.chat_history = chat_history
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_customers(db: Session, user_id: str) -> Optional[CustomersResult]:
    return db.query(CustomersResult).filter_by(user_id=user_id).first()


def get_customers_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_customers(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 — FiveCompetitionAgent → competition_results
# ─────────────────────────────────────────────────────────────────────────────

def save_competition(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> CompetitionResult:
    row = _upsert(db, CompetitionResult, user_id)
    row.direct_competitors            = data.get("direct_competitors")
    row.indirect_alternatives         = data.get("indirect_alternatives")
    row.substitute_solutions          = data.get("substitute_solutions")
    row.positioning_gaps              = data.get("positioning_gaps")
    row.porters_five_forces           = data.get("porters_five_forces")
    row.vrio_analysis                 = data.get("vrio_analysis")
    row.differentiation_opportunities = data.get("differentiation_opportunities")
    row.summary                       = data.get("summary")
    row.source_mode                   = data.get("source_mode")
    row.sources_used                  = data.get("sources_used")
    if chat_history is not None:
        row.chat_history = chat_history
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_competition(db: Session, user_id: str) -> Optional[CompetitionResult]:
    return db.query(CompetitionResult).filter_by(user_id=user_id).first()


def get_competition_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_competition(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent 6 — SixMaketPotential → market_potential_results
# ─────────────────────────────────────────────────────────────────────────────

def save_market_potential(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> MarketPotentialResult:
    row = _upsert(db, MarketPotentialResult, user_id)
    row.market_definition          = data.get("market_definition")
    row.tam                        = data.get("tam")
    row.sam                        = data.get("sam")
    row.som                        = data.get("som")
    row.market_trends              = data.get("market_trends")
    row.growth_drivers             = data.get("growth_drivers")
    row.adoption_barriers          = data.get("adoption_barriers")
    row.timing_assessment          = data.get("timing_assessment")
    row.pestel                     = data.get("pestel")
    row.opportunity_score          = data.get("opportunity_score")
    row.opportunity_attractiveness = data.get("opportunity_attractiveness")
    row.summary                    = data.get("summary")
    row.target_region              = data.get("target_region")
    row.source_mode                = data.get("source_mode")
    row.sources_used               = data.get("sources_used")
    if chat_history is not None:
        row.chat_history = chat_history
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_market_potential(db: Session, user_id: str) -> Optional[MarketPotentialResult]:
    return db.query(MarketPotentialResult).filter_by(user_id=user_id).first()


def get_market_potential_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_market_potential(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent 7 — SevenIdeaStrategy → idea_strategy_results
# ─────────────────────────────────────────────────────────────────────────────

def save_idea_strategy(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> IdeaStrategyResult:
    row = _upsert(db, IdeaStrategyResult, user_id)
    row.value_proposition        = data.get("value_proposition")
    row.positioning              = data.get("positioning")
    row.core_promise             = data.get("core_promise")
    row.differentiation_strategy = data.get("differentiation_strategy")
    row.key_assumptions          = data.get("key_assumptions")
    row.validation_priorities    = data.get("validation_priorities")
    row.strategic_direction      = data.get("strategic_direction")
    row.unfair_advantages        = data.get("unfair_advantages")
    row.strategic_risks          = data.get("strategic_risks")
    row.summary                  = data.get("summary")
    row.source_mode              = data.get("source_mode")
    row.sources_used             = data.get("sources_used")
    if chat_history is not None:
        row.chat_history = chat_history
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_idea_strategy(db: Session, user_id: str) -> Optional[IdeaStrategyResult]:
    return db.query(IdeaStrategyResult).filter_by(user_id=user_id).first()


def get_idea_strategy_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_idea_strategy(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent 8 — EightBusinessModel → business_model_results
# ─────────────────────────────────────────────────────────────────────────────

def save_business_model(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> BusinessModelResult:
    row = _upsert(db, BusinessModelResult, user_id)
    row.business_model_type    = data.get("business_model_type")
    row.business_model_canvas  = data.get("business_model_canvas")
    row.revenue_streams        = data.get("revenue_streams")
    row.pricing_strategy       = data.get("pricing_strategy")
    row.key_metrics            = data.get("key_metrics")
    row.business_model_risks   = data.get("business_model_risks")
    row.founder_fit_assessment = data.get("founder_fit_assessment")
    row.summary                = data.get("summary")
    row.source_mode            = data.get("source_mode")
    row.sources_used           = data.get("sources_used")
    if chat_history is not None:
        row.chat_history = chat_history
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_business_model(db: Session, user_id: str) -> Optional[BusinessModelResult]:
    return db.query(BusinessModelResult).filter_by(user_id=user_id).first()


def get_business_model_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_business_model(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent 9 — NineFunctionsList → functions_list_results
# ─────────────────────────────────────────────────────────────────────────────

def save_functions_list(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> FunctionsListResult:
    row = _upsert(db, FunctionsListResult, user_id)
    row.product_type                   = data.get("product_type")
    row.core_functions                 = data.get("core_functions")
    row.nice_to_have_functions         = data.get("nice_to_have_functions")
    row.future_capabilities            = data.get("future_capabilities")
    row.feature_creep_warnings         = data.get("feature_creep_warnings")
    row.function_to_pain_map           = data.get("function_to_pain_map")
    row.function_to_business_model_map = data.get("function_to_business_model_map")
    row.no_code_stack                  = data.get("no_code_stack")
    row.summary                        = data.get("summary")
    row.source_mode                    = data.get("source_mode")
    row.sources_used                   = data.get("sources_used")
    if chat_history is not None:
        row.chat_history = chat_history
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_functions_list(db: Session, user_id: str) -> Optional[FunctionsListResult]:
    return db.query(FunctionsListResult).filter_by(user_id=user_id).first()


def get_functions_list_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_functions_list(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent 10 — TenMVPPlanning → mvp_planning_results
# ─────────────────────────────────────────────────────────────────────────────

def save_mvp_planning(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> MVPPlanningResult:
    row = _upsert(db, MVPPlanningResult, user_id)
    row.mvp_goal               = data.get("mvp_goal")
    row.riskiest_assumptions   = data.get("riskiest_assumptions")
    row.scope                  = data.get("scope")
    row.core_user_flows        = data.get("core_user_flows")
    row.build_plan             = data.get("build_plan")
    row.validation_experiments = data.get("validation_experiments")
    row.launch_criteria        = data.get("launch_criteria")
    row.testing_plan           = data.get("testing_plan")
    row.qa_checklist           = data.get("qa_checklist")
    row.first_100_users_plan   = data.get("first_100_users_plan")
    row.summary                = data.get("summary")
    row.source_mode            = data.get("source_mode")
    row.sources_used           = data.get("sources_used")
    if chat_history is not None:
        row.chat_history = chat_history
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_mvp_planning(db: Session, user_id: str) -> Optional[MVPPlanningResult]:
    return db.query(MVPPlanningResult).filter_by(user_id=user_id).first()


def get_mvp_planning_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_mvp_planning(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent 11 — ElevenUnitEconomicsAgent → unit_economics_results
# ─────────────────────────────────────────────────────────────────────────────

def save_unit_economics(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> UnitEconomicsResult:
    row = _upsert(db, UnitEconomicsResult, user_id)
    row.revenue_model_summary = data.get("revenue_model_summary")
    row.pricing_assumptions   = data.get("pricing_assumptions")
    row.cost_assumptions      = data.get("cost_assumptions")
    row.gross_margin          = data.get("gross_margin")
    row.cac_analysis          = data.get("cac_analysis")
    row.ltv_analysis          = data.get("ltv_analysis")
    row.ltv_cac_ratio         = data.get("ltv_cac_ratio")
    row.payback_period        = data.get("payback_period")
    row.break_even            = data.get("break_even")
    row.monthly_projections   = data.get("monthly_projections")
    row.weak_assumptions      = data.get("weak_assumptions")
    row.pricing_tests         = data.get("pricing_tests")
    row.overall_viability     = data.get("overall_viability")
    row.summary               = data.get("summary")
    row.source_mode           = data.get("source_mode")
    row.sources_used          = data.get("sources_used")
    if chat_history is not None:
        row.chat_history = chat_history
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_unit_economics(db: Session, user_id: str) -> Optional[UnitEconomicsResult]:
    return db.query(UnitEconomicsResult).filter_by(user_id=user_id).first()


def get_unit_economics_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_unit_economics(db, user_id)
    return row.data if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent 12 — TwelveGoToMarket → go_to_market_results
# ─────────────────────────────────────────────────────────────────────────────

def save_go_to_market(
    db: Session, user_id: str, data: dict, chat_history: Optional[list] = None
) -> GoToMarketResult:
    row = _upsert(db, GoToMarketResult, user_id)
    row.target_launch_segment    = data.get("target_launch_segment")
    row.positioning_message      = data.get("positioning_message")
    row.marketing_channels       = data.get("marketing_channels")
    row.funnel_stages            = data.get("funnel_stages")
    row.launch_experiments       = data.get("launch_experiments")
    row.first_100_customers_plan = data.get("first_100_customers_plan")
    row.launch_timeline          = data.get("launch_timeline")
    row.success_metrics          = data.get("success_metrics")
    row.cac_tracking             = data.get("cac_tracking")
    row.feedback_loops           = data.get("feedback_loops")
    row.summary                  = data.get("summary")
    row.source_mode              = data.get("source_mode")
    row.sources_used             = data.get("sources_used")
    if chat_history is not None:
        row.chat_history = chat_history
    row.updated_at = datetime.utcnow()
    return _safe_commit(db, row)


def get_go_to_market(db: Session, user_id: str) -> Optional[GoToMarketResult]:
    return db.query(GoToMarketResult).filter_by(user_id=user_id).first()


def get_go_to_market_json(db: Session, user_id: str) -> Optional[dict]:
    row = get_go_to_market(db, user_id)
    return row.data if row else None
