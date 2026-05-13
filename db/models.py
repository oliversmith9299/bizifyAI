import uuid
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, JSON, Text, DateTime
from datetime import datetime

from db.connection import Base

# ─────────────────────────────────────────────────────────────────────────────
# Platform tables shared with the backend
# ─────────────────────────────────────────────────────────────────────────────

class Business(Base):
    __tablename__ = "businesses"
    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    idea_id       = Column(String, nullable=True)
    owner_id      = Column(String, nullable=False)
    industry_id   = Column(String, nullable=True)
    stage         = Column(String, nullable=True)
    context_json  = Column(JSON,   nullable=True)
    is_archived   = Column(Boolean, default=False)
    archived_at   = Column(DateTime, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Idea(Base):
    __tablename__ = "ideas"
    id                = Column(String,  primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id          = Column(String,  nullable=False)
    business_id       = Column(String,  nullable=True)
    title             = Column(String,  nullable=True)
    description       = Column(Text,    nullable=True)
    status            = Column(String,  default="draft")
    ai_score          = Column(Float,   nullable=True)
    budget            = Column(Float,   nullable=True)
    skills            = Column(JSON,    nullable=True)
    feasibility       = Column(Float,   nullable=True)
    is_score_outdated = Column(Boolean, default=False)
    is_archived       = Column(Boolean, default=False)
    archived_at       = Column(DateTime, nullable=True)
    converted_at      = Column(DateTime, nullable=True)
    created_at        = Column(DateTime, default=datetime.utcnow)
    updated_at        = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BusinessRoadmap(Base):
    __tablename__         = "business_roadmaps"
    id                    = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    business_id           = Column(String, nullable=False)
    completion_percentage = Column(Float, default=0.0)
    created_at            = Column(DateTime, default=datetime.utcnow)


class RoadmapStage(Base):
    __tablename__ = "roadmap_stages"
    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    roadmap_id    = Column(String, ForeignKey("business_roadmaps.id"), nullable=False)
    order_index   = Column(Integer, nullable=True)
    stage_type    = Column(String, nullable=True)
    status        = Column(String, default="pending")
    output_json   = Column(JSON, nullable=True)
    completed_at  = Column(DateTime, nullable=True)


class ChatSession(Base):
    __tablename__             = "chat_sessions"
    id                        = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id                   = Column(String, nullable=False)
    business_id               = Column(String, nullable=True)
    idea_id                   = Column(String, nullable=True)
    session_type              = Column(String, nullable=True)
    conversation_summary_json = Column(JSON, nullable=True)
    created_at                = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role       = Column(String, nullable=False)
    content    = Column(Text,   nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline status
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    user_id      = Column(String, primary_key=True)
    status       = Column(String, default="pending")
    current_step = Column(String, nullable=True)
    error        = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Questionnaire (raw input — keeps data blob because it is not an agent output)
# ─────────────────────────────────────────────────────────────────────────────

class QuestionnaireOutput(Base):
    __tablename__ = "questionnaire_outputs"
    user_id    = Column(String, primary_key=True)
    data       = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — OneProfileAnalysis → profile_results
# ─────────────────────────────────────────────────────────────────────────────

class ProfileResult(Base):
    __tablename__ = "profile_results"
    user_id                   = Column(String, primary_key=True)
    personality_insights      = Column(JSON, nullable=True)   # {type, motivation, traits, strengths, weaknesses}
    founder_profile           = Column(JSON, nullable=True)   # {experience_level, execution_style, risk_level, readiness, skill_level_summary, key_skill_gaps}
    recommended_industries    = Column(JSON, nullable=True)   # []
    recommended_problem_spaces = Column(JSON, nullable=True)  # []
    search_direction          = Column(JSON, nullable=True)   # {keywords: []}
    system_flags              = Column(JSON, nullable=True)   # {needs_guidance, should_suggest_learning}
    created_at                = Column(DateTime, default=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "personality_insights":       self.personality_insights       or {},
            "founder_profile":            self.founder_profile            or {},
            "recommended_industries":     self.recommended_industries     or [],
            "recommended_problem_spaces": self.recommended_problem_spaces or [],
            "search_direction":           self.search_direction           or {"keywords": []},
            "system_flags":               self.system_flags               or {},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — TwoProblemDiscovery → problems_results
# ─────────────────────────────────────────────────────────────────────────────

class ProblemsResult(Base):
    __tablename__ = "problems_results"
    user_id           = Column(String, primary_key=True)
    problems          = Column(JSON, nullable=True)   # [{id, title, description, industry, target_customer, pain_level, frequency, current_solutions, gap_opportunity, source_type, sources, evidence, validation_score}]
    customer_segments = Column(JSON, nullable=True)   # []
    personas          = Column(JSON, nullable=True)   # []
    summary_insight   = Column(Text, nullable=True)
    created_at        = Column(DateTime, default=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "problems":          self.problems          or [],
            "customer_segments": self.customer_segments or [],
            "personas":          self.personas          or [],
            "summary_insight":   self.summary_insight   or "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3a — ThreePersonalizeIdeaChat → idea_results
# ─────────────────────────────────────────────────────────────────────────────

class IdeaResult(Base):
    __tablename__ = "idea_results"
    user_id      = Column(String, primary_key=True)
    current_idea = Column(Text, nullable=True)
    chat_history = Column(JSON, default=lambda: [])
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3b — ThreeIdeaIntakeAgent → idea_intake_results
# ─────────────────────────────────────────────────────────────────────────────

class IdeaIntakeResult(Base):
    __tablename__ = "idea_intake_results"
    user_id                       = Column(String, primary_key=True)
    decision                      = Column(String, nullable=True)    # ready | needs_clarification
    idea_summary                  = Column(Text,   nullable=True)
    target_users                  = Column(JSON,   nullable=True)    # []
    industry                      = Column(String, nullable=True)
    problem_assumption            = Column(Text,   nullable=True)
    solution_assumption           = Column(Text,   nullable=True)
    business_model                = Column(String, nullable=True)
    region                        = Column(String, nullable=True)
    keywords_for_problem_discovery = Column(JSON,  nullable=True)   # []
    unclear_questions             = Column(JSON,   nullable=True)    # []
    reply                         = Column(Text,   nullable=True)
    chat_history                  = Column(JSON,   default=lambda: [])
    created_at                    = Column(DateTime, default=datetime.utcnow)
    updated_at                    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def data(self) -> dict:
        """Returns the flat intake dict that downstream pipeline helpers expect."""
        return {
            "decision":                       self.decision                       or "ready",
            "idea_summary":                   self.idea_summary                   or "",
            "target_users":                   self.target_users                   or [],
            "industry":                       self.industry                       or "",
            "problem_assumption":             self.problem_assumption             or "",
            "solution_assumption":            self.solution_assumption            or "",
            "business_model":                 self.business_model                 or "",
            "region":                         self.region                         or "Global",
            "keywords_for_problem_discovery": self.keywords_for_problem_discovery or [],
            "unclear_questions":              self.unclear_questions              or [],
            "reply":                          self.reply                          or "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — FourCustomersAgent → customers_results
# ─────────────────────────────────────────────────────────────────────────────

class CustomersResult(Base):
    __tablename__ = "customers_results"
    user_id              = Column(String,  primary_key=True)
    customer_segments    = Column(JSON,    nullable=True)   # [{id, name, description, pain_intensity, size_estimate, willingness_to_pay, why_they_care, observable_behavior, where_to_find}]
    primary_segment      = Column(JSON,    nullable=True)   # {id, reason}
    catwoe               = Column(JSON,    nullable=True)   # {customers, actors, transformation, worldview, owner, environment}
    personas             = Column(JSON,    nullable=True)   # [{name, age, job, core_pain, core_goal, quote}]
    acquisition_channels = Column(JSON,    nullable=True)   # [{channel, reason}]
    early_adopter_profile = Column(Text,   nullable=True)
    summary              = Column(Text,    nullable=True)
    source_mode          = Column(String,  nullable=True)
    sources_used         = Column(Integer, nullable=True)
    chat_history         = Column(JSON,    default=lambda: [])
    created_at           = Column(DateTime, default=datetime.utcnow)
    updated_at           = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "customer_segments":    self.customer_segments    or [],
            "primary_segment":      self.primary_segment      or {},
            "catwoe":               self.catwoe               or {},
            "personas":             self.personas             or [],
            "acquisition_channels": self.acquisition_channels or [],
            "early_adopter_profile": self.early_adopter_profile or "",
            "summary":              self.summary              or "",
            "source_mode":          self.source_mode          or "",
            "sources_used":         self.sources_used         or 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 — FiveCompetitionAgent → competition_results
# ─────────────────────────────────────────────────────────────────────────────

class CompetitionResult(Base):
    __tablename__ = "competition_results"
    user_id                      = Column(String,  primary_key=True)
    direct_competitors           = Column(JSON,    nullable=True)   # [{name, description, target_customer, pricing_model, key_features, strengths, weaknesses, market_share_estimate}]
    indirect_alternatives        = Column(JSON,    nullable=True)   # [{name, why_chosen, core_limitation}]
    substitute_solutions         = Column(JSON,    nullable=True)   # []
    positioning_gaps             = Column(JSON,    nullable=True)   # [{gap, opportunity}]
    porters_five_forces          = Column(JSON,    nullable=True)   # {bargaining_power_buyers, bargaining_power_suppliers, threat_new_entrants, threat_substitutes, competitive_rivalry}
    vrio_analysis                = Column(JSON,    nullable=True)   # [{resource, valuable, rare, inimitable, organized, conclusion}]
    differentiation_opportunities = Column(JSON,   nullable=True)   # []
    summary                      = Column(Text,    nullable=True)
    source_mode                  = Column(String,  nullable=True)
    sources_used                 = Column(Integer, nullable=True)
    chat_history                 = Column(JSON,    default=lambda: [])
    created_at                   = Column(DateTime, default=datetime.utcnow)
    updated_at                   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "direct_competitors":            self.direct_competitors            or [],
            "indirect_alternatives":         self.indirect_alternatives         or [],
            "substitute_solutions":          self.substitute_solutions          or [],
            "positioning_gaps":              self.positioning_gaps              or [],
            "porters_five_forces":           self.porters_five_forces           or {},
            "vrio_analysis":                 self.vrio_analysis                 or [],
            "differentiation_opportunities": self.differentiation_opportunities or [],
            "summary":                       self.summary                       or "",
            "source_mode":                   self.source_mode                   or "",
            "sources_used":                  self.sources_used                  or 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 6 — SixMaketPotential → market_potential_results
# ─────────────────────────────────────────────────────────────────────────────

class MarketPotentialResult(Base):
    __tablename__ = "market_potential_results"
    user_id                   = Column(String,  primary_key=True)
    market_definition         = Column(Text,    nullable=True)
    tam                       = Column(JSON,    nullable=True)   # {value, unit, source_note, methodology}
    sam                       = Column(JSON,    nullable=True)   # {value, unit, reasoning}
    som                       = Column(JSON,    nullable=True)   # {value, unit, reasoning, timeline}
    market_trends             = Column(JSON,    nullable=True)   # []
    growth_drivers            = Column(JSON,    nullable=True)   # []
    adoption_barriers         = Column(JSON,    nullable=True)   # []
    timing_assessment         = Column(JSON,    nullable=True)   # {is_right_time, reasoning}
    pestel                    = Column(JSON,    nullable=True)   # {political, economic, social, technological, environmental, legal}
    opportunity_score         = Column(Float,   nullable=True)
    opportunity_attractiveness = Column(String, nullable=True)
    summary                   = Column(Text,    nullable=True)
    target_region             = Column(String,  nullable=True)
    source_mode               = Column(String,  nullable=True)
    sources_used              = Column(Integer, nullable=True)
    chat_history              = Column(JSON,    default=lambda: [])
    created_at                = Column(DateTime, default=datetime.utcnow)
    updated_at                = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "market_definition":          self.market_definition          or "",
            "tam":                        self.tam                        or {},
            "sam":                        self.sam                        or {},
            "som":                        self.som                        or {},
            "market_trends":              self.market_trends              or [],
            "growth_drivers":             self.growth_drivers             or [],
            "adoption_barriers":          self.adoption_barriers          or [],
            "timing_assessment":          self.timing_assessment          or {},
            "pestel":                     self.pestel                     or {},
            "opportunity_score":          self.opportunity_score          or 0,
            "opportunity_attractiveness": self.opportunity_attractiveness or "",
            "summary":                    self.summary                    or "",
            "target_region":              self.target_region              or "",
            "source_mode":                self.source_mode                or "",
            "sources_used":               self.sources_used               or 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 7 — SevenIdeaStrategy → idea_strategy_results
# ─────────────────────────────────────────────────────────────────────────────

class IdeaStrategyResult(Base):
    __tablename__ = "idea_strategy_results"
    user_id                 = Column(String,  primary_key=True)
    value_proposition       = Column(JSON,    nullable=True)   # {statement, for_whom, problem_solved, key_benefit, differentiator}
    positioning             = Column(JSON,    nullable=True)   # {category, target_audience, frame_of_reference, point_of_difference, positioning_statement}
    core_promise            = Column(Text,    nullable=True)
    differentiation_strategy = Column(JSON,   nullable=True)   # {approach, key_differentiators, hard_to_copy_elements}
    key_assumptions         = Column(JSON,    nullable=True)   # [{assumption, risk_level, how_to_validate}]
    validation_priorities   = Column(JSON,    nullable=True)   # [{what_to_validate, method, success_metric, timeline}]
    strategic_direction     = Column(JSON,    nullable=True)   # {short_term_focus, medium_term_focus, long_term_vision}
    unfair_advantages       = Column(JSON,    nullable=True)   # []
    strategic_risks         = Column(JSON,    nullable=True)   # [{risk, severity, mitigation}]
    summary                 = Column(Text,    nullable=True)
    source_mode             = Column(String,  nullable=True)
    sources_used            = Column(Integer, nullable=True)
    chat_history            = Column(JSON,    default=lambda: [])
    created_at              = Column(DateTime, default=datetime.utcnow)
    updated_at              = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "value_proposition":        self.value_proposition        or {},
            "positioning":              self.positioning              or {},
            "core_promise":             self.core_promise             or "",
            "differentiation_strategy": self.differentiation_strategy or {},
            "key_assumptions":          self.key_assumptions          or [],
            "validation_priorities":    self.validation_priorities    or [],
            "strategic_direction":      self.strategic_direction      or {},
            "unfair_advantages":        self.unfair_advantages        or [],
            "strategic_risks":          self.strategic_risks          or [],
            "summary":                  self.summary                  or "",
            "source_mode":              self.source_mode              or "",
            "sources_used":             self.sources_used             or 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 8 — EightBusinessModel → business_model_results
# ─────────────────────────────────────────────────────────────────────────────

class BusinessModelResult(Base):
    __tablename__ = "business_model_results"
    user_id                = Column(String,  primary_key=True)
    business_model_type    = Column(String,  nullable=True)
    business_model_canvas  = Column(JSON,    nullable=True)   # {customer_segments, value_propositions, channels, customer_relationships, revenue_streams, key_resources, key_activities, key_partnerships, cost_structure}
    revenue_streams        = Column(JSON,    nullable=True)   # [{id, name, type, description, pricing, estimated_monthly_at_scale}]
    pricing_strategy       = Column(JSON,    nullable=True)   # {approach, rationale, price_points, discounting_policy, price_sensitivity_note}
    key_metrics            = Column(JSON,    nullable=True)   # []
    business_model_risks   = Column(JSON,    nullable=True)   # [{risk, severity, mitigation}]
    founder_fit_assessment = Column(JSON,    nullable=True)   # {can_execute, reasoning, biggest_execution_risk}
    summary                = Column(Text,    nullable=True)
    source_mode            = Column(String,  nullable=True)
    sources_used           = Column(Integer, nullable=True)
    chat_history           = Column(JSON,    default=lambda: [])
    created_at             = Column(DateTime, default=datetime.utcnow)
    updated_at             = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "business_model_type":    self.business_model_type    or "",
            "business_model_canvas":  self.business_model_canvas  or {},
            "revenue_streams":        self.revenue_streams        or [],
            "pricing_strategy":       self.pricing_strategy       or {},
            "key_metrics":            self.key_metrics            or [],
            "business_model_risks":   self.business_model_risks   or [],
            "founder_fit_assessment": self.founder_fit_assessment or {},
            "summary":                self.summary                or "",
            "source_mode":            self.source_mode            or "",
            "sources_used":           self.sources_used           or 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 9 — NineFunctionsList → functions_list_results
# ─────────────────────────────────────────────────────────────────────────────

class FunctionsListResult(Base):
    __tablename__ = "functions_list_results"
    user_id                      = Column(String,  primary_key=True)
    product_type                 = Column(String,  nullable=True)
    core_functions               = Column(JSON,    nullable=True)   # [{id, name, description, why_needed, priority, complexity, no_code_solution}]
    nice_to_have_functions       = Column(JSON,    nullable=True)   # [{id, name, description, when_to_add, trigger}]
    future_capabilities          = Column(JSON,    nullable=True)   # [{id, name, description, vision}]
    feature_creep_warnings       = Column(JSON,    nullable=True)   # []
    function_to_pain_map         = Column(JSON,    nullable=True)   # [{function_id, pain_point, how_it_solves}]
    function_to_business_model_map = Column(JSON,  nullable=True)   # [{function_id, business_need, connection}]
    no_code_stack                = Column(JSON,    nullable=True)   # [{tool, purpose, monthly_cost_usd}]
    summary                      = Column(Text,    nullable=True)
    source_mode                  = Column(String,  nullable=True)
    sources_used                 = Column(Integer, nullable=True)
    chat_history                 = Column(JSON,    default=lambda: [])
    created_at                   = Column(DateTime, default=datetime.utcnow)
    updated_at                   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "product_type":                   self.product_type                   or "",
            "core_functions":                 self.core_functions                 or [],
            "nice_to_have_functions":         self.nice_to_have_functions         or [],
            "future_capabilities":            self.future_capabilities            or [],
            "feature_creep_warnings":         self.feature_creep_warnings         or [],
            "function_to_pain_map":           self.function_to_pain_map           or [],
            "function_to_business_model_map": self.function_to_business_model_map or [],
            "no_code_stack":                  self.no_code_stack                  or [],
            "summary":                        self.summary                        or "",
            "source_mode":                    self.source_mode                    or "",
            "sources_used":                   self.sources_used                   or 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 10 — TenMVPPlanning → mvp_planning_results
# ─────────────────────────────────────────────────────────────────────────────

class MVPPlanningResult(Base):
    __tablename__ = "mvp_planning_results"
    user_id               = Column(String,  primary_key=True)
    mvp_goal              = Column(Text,    nullable=True)
    riskiest_assumptions  = Column(JSON,    nullable=True)   # [{id, assumption, risk_level, validation_method, kill_signal}]
    scope                 = Column(JSON,    nullable=True)   # {included: [], excluded: []}
    core_user_flows       = Column(JSON,    nullable=True)   # [{id, name, steps, success_metric}]
    build_plan            = Column(JSON,    nullable=True)   # {phases: [{name, tasks, milestone}], total_timeline, no_code_tools_needed}
    validation_experiments = Column(JSON,   nullable=True)   # [{id, hypothesis, method, success_metric, timeline, cost_usd}]
    launch_criteria       = Column(JSON,    nullable=True)   # {must_be_true, success_metrics, kill_criteria}
    testing_plan          = Column(JSON,    nullable=True)   # [{area, method, pass_criteria}]
    qa_checklist          = Column(JSON,    nullable=True)   # []
    first_100_users_plan  = Column(Text,    nullable=True)
    summary               = Column(Text,    nullable=True)
    source_mode           = Column(String,  nullable=True)
    sources_used          = Column(Integer, nullable=True)
    chat_history          = Column(JSON,    default=lambda: [])
    created_at            = Column(DateTime, default=datetime.utcnow)
    updated_at            = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "mvp_goal":               self.mvp_goal               or "",
            "riskiest_assumptions":   self.riskiest_assumptions   or [],
            "scope":                  self.scope                  or {},
            "core_user_flows":        self.core_user_flows        or [],
            "build_plan":             self.build_plan             or {},
            "validation_experiments": self.validation_experiments or [],
            "launch_criteria":        self.launch_criteria        or {},
            "testing_plan":           self.testing_plan           or [],
            "qa_checklist":           self.qa_checklist           or [],
            "first_100_users_plan":   self.first_100_users_plan   or "",
            "summary":                self.summary                or "",
            "source_mode":            self.source_mode            or "",
            "sources_used":           self.sources_used           or 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 11 — ElevenUnitEconomicsAgent → unit_economics_results
# ─────────────────────────────────────────────────────────────────────────────

class UnitEconomicsResult(Base):
    __tablename__ = "unit_economics_results"
    user_id               = Column(String,  primary_key=True)
    revenue_model_summary = Column(Text,    nullable=True)
    pricing_assumptions   = Column(JSON,    nullable=True)   # {average_order_value_usd, commission_rate_pct, avg_monthly_orders_per_buyer, avg_monthly_active_buyers_month_6, avg_seller_subscription_take_rate_pct}
    cost_assumptions      = Column(JSON,    nullable=True)   # {fixed_monthly_usd, variable_cost_pct_of_gmv, cac_paid_usd, cac_organic_usd, organic_to_paid_ratio, avg_blended_cac_usd}
    gross_margin          = Column(JSON,    nullable=True)   # {revenue_per_transaction_usd, direct_cost_per_transaction_usd, gross_profit_per_transaction_usd, gross_margin_pct, note}
    cac_analysis          = Column(JSON,    nullable=True)   # {blended_cac_usd, paid_cac_usd, organic_cac_usd, primary_acquisition_channel, cac_benchmark_comparison, cac_trend}
    ltv_analysis          = Column(JSON,    nullable=True)   # {avg_customer_lifespan_months, avg_monthly_revenue_per_buyer_usd, ltv_usd, churn_rate_monthly_pct, churn_assumption_note}
    ltv_cac_ratio         = Column(JSON,    nullable=True)   # {ratio, interpretation, target_ratio, is_viable}
    payback_period        = Column(JSON,    nullable=True)   # {months, interpretation, calculation}
    break_even            = Column(JSON,    nullable=True)   # {monthly_fixed_cost_usd, revenue_per_buyer_per_month_usd, buyers_needed_to_break_even, gmv_needed_to_break_even_usd, timeline_to_break_even, note}
    monthly_projections   = Column(JSON,    nullable=True)   # [{month, active_buyers, gmv_usd, revenue_usd, costs_usd, profit_usd}]
    weak_assumptions      = Column(JSON,    nullable=True)   # [{assumption, risk, impact_if_wrong, test}]
    pricing_tests         = Column(JSON,    nullable=True)   # [{test, hypothesis, method, success_metric}]
    overall_viability     = Column(JSON,    nullable=True)   # {is_economically_viable, verdict, key_risks, recommended_actions}
    summary               = Column(Text,    nullable=True)
    source_mode           = Column(String,  nullable=True)
    sources_used          = Column(Integer, nullable=True)
    chat_history          = Column(JSON,    default=lambda: [])
    created_at            = Column(DateTime, default=datetime.utcnow)
    updated_at            = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "revenue_model_summary": self.revenue_model_summary or "",
            "pricing_assumptions":   self.pricing_assumptions   or {},
            "cost_assumptions":      self.cost_assumptions      or {},
            "gross_margin":          self.gross_margin          or {},
            "cac_analysis":          self.cac_analysis          or {},
            "ltv_analysis":          self.ltv_analysis          or {},
            "ltv_cac_ratio":         self.ltv_cac_ratio         or {},
            "payback_period":        self.payback_period        or {},
            "break_even":            self.break_even            or {},
            "monthly_projections":   self.monthly_projections   or [],
            "weak_assumptions":      self.weak_assumptions      or [],
            "pricing_tests":         self.pricing_tests         or [],
            "overall_viability":     self.overall_viability     or {},
            "summary":               self.summary               or "",
            "source_mode":           self.source_mode           or "",
            "sources_used":          self.sources_used          or 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 12 — TwelveGoToMarket → go_to_market_results
# ─────────────────────────────────────────────────────────────────────────────

class GoToMarketResult(Base):
    __tablename__ = "go_to_market_results"
    user_id                   = Column(String,  primary_key=True)
    target_launch_segment     = Column(JSON,    nullable=True)   # {segment_name, why_first, size_estimate, beachhead_cities}
    positioning_message       = Column(JSON,    nullable=True)   # {headline, subheadline, proof_points, tone, arabic_headline}
    marketing_channels        = Column(JSON,    nullable=True)   # [{channel, role, weekly_effort_hours, monthly_cost_usd, target_metric, content_types, is_paid}]
    funnel_stages             = Column(JSON,    nullable=True)   # [{stage, description, channels, key_action, conversion_target_pct, metric}]
    launch_experiments        = Column(JSON,    nullable=True)   # [{id, name, hypothesis, method, success_metric, timeline, budget_usd}]
    first_100_customers_plan  = Column(JSON,    nullable=True)   # {target_timeline, weekly_breakdown, total_estimated_budget_usd, key_actions}
    launch_timeline           = Column(JSON,    nullable=True)   # [{week, focus, goal}]
    success_metrics           = Column(JSON,    nullable=True)   # {week_4_targets, week_8_targets, kill_metrics}
    cac_tracking              = Column(JSON,    nullable=True)   # {tracking_method, by_channel, weekly_review, blended_cac_formula}
    feedback_loops            = Column(JSON,    nullable=True)   # [{trigger, method, what_to_capture, action}]
    summary                   = Column(Text,    nullable=True)
    source_mode               = Column(String,  nullable=True)
    sources_used              = Column(Integer, nullable=True)
    chat_history              = Column(JSON,    default=lambda: [])
    created_at                = Column(DateTime, default=datetime.utcnow)
    updated_at                = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def data(self) -> dict:
        return {
            "target_launch_segment":    self.target_launch_segment    or {},
            "positioning_message":      self.positioning_message      or {},
            "marketing_channels":       self.marketing_channels       or [],
            "funnel_stages":            self.funnel_stages            or [],
            "launch_experiments":       self.launch_experiments       or [],
            "first_100_customers_plan": self.first_100_customers_plan or {},
            "launch_timeline":          self.launch_timeline          or [],
            "success_metrics":          self.success_metrics          or {},
            "cac_tracking":             self.cac_tracking             or {},
            "feedback_loops":           self.feedback_loops           or [],
            "summary":                  self.summary                  or "",
            "source_mode":              self.source_mode              or "",
            "sources_used":             self.sources_used             or 0,
        }
