"""
Unit tests for db/models.py
=============================
Verifies that each model's `.data` property returns safe defaults
(never None, always the correct type) when all columns are NULL.
No DB connection required — instances are created directly in Python.
"""

from db.models import (
    BusinessModelResult,
    CompetitionResult,
    CustomersResult,
    FunctionsListResult,
    GoToMarketResult,
    IdeaIntakeResult,
    IdeaStrategyResult,
    MarketPotentialResult,
    MVPPlanningResult,
    ProfileResult,
    ProblemsResult,
    UnitEconomicsResult,
)


def _blank(model_cls):
    """Return a properly initialised ORM instance with all nullable columns set to None."""
    obj = model_cls()
    for col in model_cls.__table__.columns:
        if col.nullable:
            setattr(obj, col.name, None)
    return obj


# ── CustomersResult ───────────────────────────────────────────────────────────

class TestCustomersResultData:
    def test_customer_segments_defaults_to_empty_list(self):
        row = _blank(CustomersResult)
        assert row.data["customer_segments"] == []

    def test_primary_segment_defaults_to_empty_dict(self):
        row = _blank(CustomersResult)
        assert row.data["primary_segment"] == {}

    def test_personas_defaults_to_empty_list(self):
        row = _blank(CustomersResult)
        assert row.data["personas"] == []

    def test_summary_defaults_to_empty_string(self):
        row = _blank(CustomersResult)
        assert row.data["summary"] == ""

    def test_sources_used_defaults_to_zero(self):
        row = _blank(CustomersResult)
        assert row.data["sources_used"] == 0

    def test_data_contains_all_expected_keys(self):
        row = _blank(CustomersResult)
        expected = {
            "customer_segments", "primary_segment", "catwoe", "personas",
            "acquisition_channels", "early_adopter_profile", "summary",
            "source_mode", "sources_used", "sources_list",
        }
        assert expected == set(row.data.keys())


# ── CompetitionResult ─────────────────────────────────────────────────────────

class TestCompetitionResultData:
    def test_direct_competitors_defaults_to_empty_list(self):
        row = _blank(CompetitionResult)
        assert row.data["direct_competitors"] == []

    def test_porters_five_forces_defaults_to_empty_dict(self):
        row = _blank(CompetitionResult)
        assert row.data["porters_five_forces"] == {}

    def test_vrio_analysis_defaults_to_empty_list(self):
        row = _blank(CompetitionResult)
        assert row.data["vrio_analysis"] == []

    def test_summary_defaults_to_empty_string(self):
        row = _blank(CompetitionResult)
        assert row.data["summary"] == ""


# ── MarketPotentialResult ─────────────────────────────────────────────────────

class TestMarketPotentialResultData:
    def test_tam_defaults_to_empty_dict(self):
        row = _blank(MarketPotentialResult)
        assert row.data["tam"] == {}

    def test_market_trends_defaults_to_empty_list(self):
        row = _blank(MarketPotentialResult)
        assert row.data["market_trends"] == []

    def test_opportunity_score_defaults_to_zero(self):
        row = _blank(MarketPotentialResult)
        assert row.data["opportunity_score"] == 0

    def test_market_definition_defaults_to_empty_string(self):
        row = _blank(MarketPotentialResult)
        assert row.data["market_definition"] == ""


# ── IdeaStrategyResult ────────────────────────────────────────────────────────

class TestIdeaStrategyResultData:
    def test_value_proposition_defaults_to_empty_dict(self):
        row = _blank(IdeaStrategyResult)
        assert row.data["value_proposition"] == {}

    def test_key_assumptions_defaults_to_empty_list(self):
        row = _blank(IdeaStrategyResult)
        assert row.data["key_assumptions"] == []

    def test_unfair_advantages_defaults_to_empty_list(self):
        row = _blank(IdeaStrategyResult)
        assert row.data["unfair_advantages"] == []


# ── BusinessModelResult ───────────────────────────────────────────────────────

class TestBusinessModelResultData:
    def test_business_model_type_defaults_to_empty_string(self):
        row = _blank(BusinessModelResult)
        assert row.data["business_model_type"] == ""

    def test_revenue_streams_defaults_to_empty_list(self):
        row = _blank(BusinessModelResult)
        assert row.data["revenue_streams"] == []

    def test_business_model_canvas_defaults_to_empty_dict(self):
        row = _blank(BusinessModelResult)
        assert row.data["business_model_canvas"] == {}


# ── FunctionsListResult ───────────────────────────────────────────────────────

class TestFunctionsListResultData:
    def test_core_functions_defaults_to_empty_list(self):
        row = _blank(FunctionsListResult)
        assert row.data["core_functions"] == []

    def test_product_type_defaults_to_empty_string(self):
        row = _blank(FunctionsListResult)
        assert row.data["product_type"] == ""

    def test_no_code_stack_defaults_to_empty_list(self):
        row = _blank(FunctionsListResult)
        assert row.data["no_code_stack"] == []


# ── IdeaIntakeResult ──────────────────────────────────────────────────────────

class TestIdeaIntakeResultData:
    def test_decision_defaults_to_ready(self):
        row = _blank(IdeaIntakeResult)
        assert row.data["decision"] == "ready"

    def test_region_defaults_to_global(self):
        row = _blank(IdeaIntakeResult)
        assert row.data["region"] == "Global"

    def test_target_users_defaults_to_empty_list(self):
        row = _blank(IdeaIntakeResult)
        assert row.data["target_users"] == []


# ── ProfileResult ─────────────────────────────────────────────────────────────

class TestProfileResultData:
    def test_personality_insights_defaults_to_empty_dict(self):
        row = _blank(ProfileResult)
        assert row.data["personality_insights"] == {}

    def test_recommended_industries_defaults_to_empty_list(self):
        row = _blank(ProfileResult)
        assert row.data["recommended_industries"] == []

    def test_search_direction_defaults_to_keywords_dict(self):
        row = _blank(ProfileResult)
        assert row.data["search_direction"] == {"keywords": []}


# ── ProblemsResult ────────────────────────────────────────────────────────────

class TestProblemsResultData:
    def test_problems_defaults_to_empty_list(self):
        row = _blank(ProblemsResult)
        assert row.data["problems"] == []

    def test_summary_insight_defaults_to_empty_string(self):
        row = _blank(ProblemsResult)
        assert row.data["summary_insight"] == ""
