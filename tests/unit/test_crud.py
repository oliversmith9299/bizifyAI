"""
Unit tests for db/crud.py
===========================
All tests run against the SQLite in-memory database configured in conftest.py.
Each test gets a fresh, rolled-back session so tests are fully isolated.
"""

import pytest

from db import crud
from db.models import (
    CustomersResult,
    CompetitionResult,
    MarketPotentialResult,
    PipelineRun,
    QuestionnaireOutput,
    IdeaResult,
    IdeaIntakeResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline status
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineStatus:
    def test_upsert_creates_new_run(self, db_session):
        run = crud.upsert_pipeline_status(db_session, "user-1", "running", step="profile_analysis")
        assert run.user_id == "user-1"
        assert run.status == "running"
        assert run.current_step == "profile_analysis"

    def test_upsert_updates_existing_run(self, db_session):
        crud.upsert_pipeline_status(db_session, "user-2", "running")
        updated = crud.upsert_pipeline_status(db_session, "user-2", "done", step="go_to_market")
        assert updated.status == "done"
        assert updated.current_step == "go_to_market"

    def test_upsert_stores_error(self, db_session):
        run = crud.upsert_pipeline_status(db_session, "user-3", "error", error="Agent crashed")
        assert run.error == "Agent crashed"

    def test_get_pipeline_status_returns_none_when_missing(self, db_session):
        result = crud.get_pipeline_status(db_session, "nonexistent-user")
        assert result is None

    def test_get_pipeline_status_returns_row(self, db_session):
        crud.upsert_pipeline_status(db_session, "user-4", "pending")
        row = crud.get_pipeline_status(db_session, "user-4")
        assert row is not None
        assert row.status == "pending"


# ─────────────────────────────────────────────────────────────────────────────
# Questionnaire output
# ─────────────────────────────────────────────────────────────────────────────

class TestQuestionnaireOutput:
    def test_save_and_get_questionnaire_output(self, db_session):
        data = {"name": "Test User", "industry": "Tech"}
        crud.save_questionnaire_output(db_session, "user-10", data)
        row = crud.get_questionnaire_output(db_session, "user-10")
        assert row is not None
        assert row.data["name"] == "Test User"

    def test_get_questionnaire_output_json_returns_none_when_missing(self, db_session):
        result = crud.get_questionnaire_output_json(db_session, "ghost-user")
        assert result is None

    def test_get_questionnaire_output_json_returns_data_dict(self, db_session):
        crud.save_questionnaire_output(db_session, "user-11", {"key": "value"})
        result = crud.get_questionnaire_output_json(db_session, "user-11")
        assert result == {"key": "value"}

    def test_upsert_overwrites_existing_questionnaire(self, db_session):
        crud.save_questionnaire_output(db_session, "user-12", {"v": 1})
        crud.save_questionnaire_output(db_session, "user-12", {"v": 2})
        result = crud.get_questionnaire_output_json(db_session, "user-12")
        assert result["v"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Customers
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CUSTOMERS = {
    "customer_segments":    [{"id": "CS1", "name": "Shoppers"}],
    "primary_segment":      {"id": "CS1", "reason": "Biggest pain"},
    "catwoe_analysis":      {"customers": "Shoppers"},
    "personas":             [{"name": "Amira", "age": "28"}],
    "acquisition_channels": ["Social media"],
    "early_adopter_profile": "Young, tech-savvy",
    "summary":              "Focus on CS1",
    "source_mode":          "web_sourced",
    "sources_used":         10,
    "sources_list":         [{"url": "https://example.com", "title": "Example"}],
}


class TestCustomersCRUD:
    def test_save_customers_creates_row(self, db_session):
        row = crud.save_customers(db_session, "user-20", SAMPLE_CUSTOMERS)
        assert row.user_id == "user-20"
        assert row.summary == "Focus on CS1"
        assert row.sources_used == 10

    def test_save_customers_with_chat_history(self, db_session):
        history = [{"role": "user", "content": "explain CS1"}]
        crud.save_customers(db_session, "user-21", SAMPLE_CUSTOMERS, chat_history=history)
        row = crud.get_customers(db_session, "user-21")
        assert row.chat_history == history

    def test_save_customers_updates_existing(self, db_session):
        crud.save_customers(db_session, "user-22", SAMPLE_CUSTOMERS)
        updated = {**SAMPLE_CUSTOMERS, "summary": "Updated summary"}
        crud.save_customers(db_session, "user-22", updated)
        row = crud.get_customers(db_session, "user-22")
        assert row.summary == "Updated summary"

    def test_get_customers_returns_none_when_missing(self, db_session):
        assert crud.get_customers(db_session, "nobody") is None

    def test_get_customers_json_returns_none_when_missing(self, db_session):
        assert crud.get_customers_json(db_session, "nobody") is None

    def test_get_customers_json_returns_data_property(self, db_session):
        crud.save_customers(db_session, "user-23", SAMPLE_CUSTOMERS)
        data = crud.get_customers_json(db_session, "user-23")
        assert isinstance(data, dict)
        assert data["summary"] == "Focus on CS1"

    def test_catwoe_stored_from_catwoe_analysis_key(self, db_session):
        payload = {**SAMPLE_CUSTOMERS, "catwoe_analysis": {"customers": "All"}}
        del payload["catwoe_analysis"]
        payload["catwoe_analysis"] = {"customers": "All"}
        crud.save_customers(db_session, "user-24", payload)
        row = crud.get_customers(db_session, "user-24")
        assert row.catwoe == {"customers": "All"}


# ─────────────────────────────────────────────────────────────────────────────
# Competition
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_COMPETITION = {
    "direct_competitors":            [{"name": "Souq", "description": "..."}],
    "indirect_alternatives":         [{"name": "Social media"}],
    "substitute_solutions":          ["DIY products"],
    "positioning_gaps":              [{"gap": "trust", "opportunity": "build trust"}],
    "porters_five_forces":           {"competitive_rivalry": {"level": "high"}},
    "vrio_analysis":                 [{"resource": "brand", "conclusion": "advantage"}],
    "differentiation_opportunities": ["personalization"],
    "summary":                       "High competition",
    "source_mode":                   "web_sourced",
    "sources_used":                  12,
    "sources_list":                  [],
}


class TestCompetitionCRUD:
    def test_save_and_get_competition(self, db_session):
        crud.save_competition(db_session, "user-30", SAMPLE_COMPETITION)
        row = crud.get_competition(db_session, "user-30")
        assert row is not None
        assert row.summary == "High competition"
        assert row.sources_used == 12

    def test_get_competition_json_returns_none_when_missing(self, db_session):
        assert crud.get_competition_json(db_session, "nobody") is None

    def test_get_competition_json_returns_data_dict(self, db_session):
        crud.save_competition(db_session, "user-31", SAMPLE_COMPETITION)
        data = crud.get_competition_json(db_session, "user-31")
        assert data["porters_five_forces"]["competitive_rivalry"]["level"] == "high"

    def test_save_competition_preserves_chat_history(self, db_session):
        history = [{"role": "assistant", "content": "Here is analysis..."}]
        crud.save_competition(db_session, "user-32", SAMPLE_COMPETITION, chat_history=history)
        row = crud.get_competition(db_session, "user-32")
        assert row.chat_history == history


# ─────────────────────────────────────────────────────────────────────────────
# Market potential
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_MARKET = {
    "market_definition":          "Global e-commerce",
    "target_region":              "MENA",
    "tam":                        {"value": "6.3 trillion", "unit": "USD"},
    "sam":                        {"value": "150 billion",  "unit": "USD"},
    "som":                        {"value": "10 million",   "unit": "USD"},
    "market_trends":              ["Mobile adoption"],
    "growth_drivers":             ["Internet access"],
    "adoption_barriers":          ["Trust issues"],
    "timing_assessment":          {"is_right_time": True},
    "pestel":                     {"political": {"impact": "neutral"}},
    "opportunity_score":          7.0,
    "opportunity_attractiveness": "medium",
    "summary":                    "Good opportunity",
    "source_mode":                "web_sourced",
    "sources_used":               12,
    "sources_list":               [],
}


class TestMarketPotentialCRUD:
    def test_save_and_get_market_potential(self, db_session):
        crud.save_market_potential(db_session, "user-40", SAMPLE_MARKET)
        row = crud.get_market_potential(db_session, "user-40")
        assert row is not None
        assert row.opportunity_score == 7.0
        assert row.target_region == "MENA"

    def test_get_market_potential_json_returns_none_when_missing(self, db_session):
        assert crud.get_market_potential_json(db_session, "nobody") is None

    def test_get_market_potential_json_returns_full_data(self, db_session):
        crud.save_market_potential(db_session, "user-41", SAMPLE_MARKET)
        data = crud.get_market_potential_json(db_session, "user-41")
        assert data["tam"]["value"] == "6.3 trillion"
        assert data["opportunity_attractiveness"] == "medium"


# ─────────────────────────────────────────────────────────────────────────────
# Idea
# ─────────────────────────────────────────────────────────────────────────────

class TestIdeaCRUD:
    def test_save_and_get_idea(self, db_session):
        crud.save_idea(db_session, "user-50", "MENA marketplace", [])
        row = crud.get_idea(db_session, "user-50")
        assert row is not None
        assert row.current_idea == "MENA marketplace"

    def test_save_idea_updates_existing(self, db_session):
        crud.save_idea(db_session, "user-51", "idea v1", [])
        crud.save_idea(db_session, "user-51", "idea v2", [{"role": "user", "content": "refine"}])
        row = crud.get_idea(db_session, "user-51")
        assert row.current_idea == "idea v2"

    def test_get_idea_returns_none_when_missing(self, db_session):
        assert crud.get_idea(db_session, "nobody") is None


# ─────────────────────────────────────────────────────────────────────────────
# Idea intake
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_INTAKE = {
    "decision":           "ready",
    "idea_summary":       "E-commerce marketplace",
    "target_users":       ["Young shoppers"],
    "industry":           "E-commerce",
    "problem_assumption": "Hard to find authentic products",
    "solution_assumption":"A trustworthy marketplace",
    "business_model":     "Commission",
    "region":             "MENA",
    "keywords_for_problem_discovery": ["marketplace", "MENA"],
    "unclear_questions":  [],
    "reply":              "Great idea!",
}


class TestIdeaIntakeCRUD:
    def test_save_and_get_idea_intake(self, db_session):
        crud.save_idea_intake(db_session, "user-60", SAMPLE_INTAKE)
        row = crud.get_idea_intake(db_session, "user-60")
        assert row is not None
        assert row.idea_summary == "E-commerce marketplace"
        assert row.region == "MENA"

    def test_get_idea_intake_json_returns_none_when_missing(self, db_session):
        assert crud.get_idea_intake_json(db_session, "nobody") is None

    def test_get_idea_intake_json_returns_data_property(self, db_session):
        crud.save_idea_intake(db_session, "user-61", SAMPLE_INTAKE)
        data = crud.get_idea_intake_json(db_session, "user-61")
        assert data["decision"] == "ready"
        assert data["region"] == "MENA"
