"""
Integration tests for /pipeline/customers endpoints.

External calls (agent LLM + Serper) are mocked so tests run without API keys.
Agents are imported inside route function bodies, so we patch at the agent
module level (agents.FourCustomersAgent.*), not at the route module level.
"""

from unittest.mock import patch

import pytest

from db import crud

AGENT_RESULT = {
    "customer_segments":     [{"id": "CS1", "name": "Young Shoppers"}],
    "primary_segment":       {"id": "CS1", "reason": "High pain"},
    "catwoe_analysis":       {"customers": "Shoppers"},
    "personas":              [{"name": "Amira", "age": "28"}],
    "acquisition_channels":  ["Social media"],
    "early_adopter_profile": "Tech-savvy youth",
    "summary":               "Target CS1",
    "source_mode":           "web_sourced",
    "sources_used":          10,
    "sources_list":          [],
}


def _seed_idea(db_session, user_id: str):
    crud.save_idea(db_session, user_id, "MENA marketplace for authentic products", [])


def _seed_problems(db_session, user_id: str):
    crud.save_problems(db_session, user_id, {
        "problems":          [{"id": "P1", "title": "Hard to find authentic goods"}],
        "customer_segments": [],
        "personas":          [],
        "summary_insight":   "Real pain",
    })


def _seed_customers(db_session, user_id: str):
    crud.save_customers(db_session, user_id, AGENT_RESULT)


# ── GET /pipeline/customers/{user_id} ─────────────────────────────────────────

class TestGetCustomers:
    def test_not_found_returns_404(self, client, api_headers):
        resp = client.get("/pipeline/customers/unknown-user", headers=api_headers)
        assert resp.status_code == 404

    def test_returns_customers_data_and_chat_history(self, client, api_headers, db_session):
        _seed_customers(db_session, "u-get-1")
        resp = client.get("/pipeline/customers/u-get-1", headers=api_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "u-get-1"
        assert "customers" in body
        assert "chat_history" in body

    def test_returned_customers_match_stored_data(self, client, api_headers, db_session):
        _seed_customers(db_session, "u-get-2")
        resp = client.get("/pipeline/customers/u-get-2", headers=api_headers)
        customers = resp.json()["customers"]
        assert customers["summary"] == "Target CS1"
        assert customers["sources_used"] == 10


# ── POST /pipeline/customers/{user_id} ────────────────────────────────────────

class TestPostCustomers:
    def test_missing_idea_returns_425(self, client, api_headers):
        resp = client.post("/pipeline/customers/no-idea-user", headers=api_headers)
        assert resp.status_code == 425

    def test_missing_problems_returns_425(self, client, api_headers, db_session):
        _seed_idea(db_session, "u-no-problems")
        resp = client.post("/pipeline/customers/u-no-problems", headers=api_headers)
        assert resp.status_code == 425

    def test_success_returns_done_status(self, client, api_headers, db_session):
        _seed_idea(db_session, "u-post-ok")
        _seed_problems(db_session, "u-post-ok")

        with patch(
            "agents.FourCustomersAgent.run_customers_analysis",
            return_value=AGENT_RESULT,
        ):
            resp = client.post("/pipeline/customers/u-post-ok", headers=api_headers)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "done"
        assert body["user_id"] == "u-post-ok"

    def test_success_response_contains_customers_key(self, client, api_headers, db_session):
        _seed_idea(db_session, "u-post-ok2")
        _seed_problems(db_session, "u-post-ok2")

        with patch(
            "agents.FourCustomersAgent.run_customers_analysis",
            return_value=AGENT_RESULT,
        ):
            resp = client.post("/pipeline/customers/u-post-ok2", headers=api_headers)

        assert "customers" in resp.json()


# ── POST /pipeline/customers/{user_id}/chat ────────────────────────────────────

class TestChatCustomers:
    def test_no_analysis_returns_425(self, client, api_headers):
        payload = {"user_id": "no-analysis-user", "message": "tell me more", "history": []}
        resp = client.post(
            "/pipeline/customers/no-analysis-user/chat",
            json=payload,
            headers=api_headers,
        )
        assert resp.status_code == 425

    def test_chat_returns_reply(self, client, api_headers, db_session):
        _seed_customers(db_session, "u-chat-1")

        with patch(
            "agents.FourCustomersAgent.chat_customers",
            return_value="CS1 is your primary segment.",
        ):
            payload = {"user_id": "u-chat-1", "message": "explain CS1", "history": []}
            resp = client.post(
                "/pipeline/customers/u-chat-1/chat",
                json=payload,
                headers=api_headers,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["reply"] == "CS1 is your primary segment."
        assert body["user_id"] == "u-chat-1"

    def test_chat_increments_history_length(self, client, api_headers, db_session):
        _seed_customers(db_session, "u-chat-2")

        with patch("agents.FourCustomersAgent.chat_customers", return_value="Response"):
            payload = {
                "user_id": "u-chat-2",
                "message": "hello",
                "history": [{"role": "user", "content": "prior"}],
            }
            resp = client.post(
                "/pipeline/customers/u-chat-2/chat",
                json=payload,
                headers=api_headers,
            )

        # prior history (1) + user message (1) + assistant reply (1) = 3
        assert resp.json()["chat_history_length"] == 3


# ── POST /pipeline/customers/{user_id}/regenerate ─────────────────────────────

class TestRegenerateCustomers:
    def test_regenerate_without_idea_returns_425(self, client, api_headers):
        resp = client.post("/pipeline/customers/regen-user/regenerate", headers=api_headers)
        assert resp.status_code == 425

    def test_regenerate_with_data_returns_done(self, client, api_headers, db_session):
        _seed_idea(db_session, "u-regen")
        _seed_problems(db_session, "u-regen")

        with patch(
            "agents.FourCustomersAgent.run_customers_analysis",
            return_value=AGENT_RESULT,
        ):
            resp = client.post("/pipeline/customers/u-regen/regenerate", headers=api_headers)

        assert resp.status_code == 200
        assert resp.json()["status"] == "done"
