"""
Integration tests for /pipeline/competition endpoints.
"""

from unittest.mock import patch

import pytest

from db import crud

AGENT_RESULT = {
    "direct_competitors":            [{"id": "DC1", "name": "Souq"}],
    "indirect_alternatives":         [{"id": "IA1", "name": "Social media"}],
    "substitute_solutions":          ["DIY products"],
    "positioning_gaps":              [{"gap": "trust", "opportunity": "build trust"}],
    "porters_five_forces":           {"competitive_rivalry": {"level": "high"}},
    "vrio_analysis":                 [],
    "differentiation_opportunities": ["personalization"],
    "summary":                       "Competitive landscape",
    "source_mode":                   "web_sourced",
    "sources_used":                  12,
    "sources_list":                  [],
}


def _seed_idea(db_session, uid):
    crud.save_idea(db_session, uid, "Marketplace idea", [])


def _seed_problems(db_session, uid):
    crud.save_problems(db_session, uid, {
        "problems":          [],
        "customer_segments": [],
        "personas":          [],
        "summary_insight":   "",
    })


def _seed_competition(db_session, uid):
    crud.save_competition(db_session, uid, AGENT_RESULT)


# ── GET /pipeline/competition/{user_id} ───────────────────────────────────────

class TestGetCompetition:
    def test_not_found_returns_404(self, client, api_headers):
        resp = client.get("/pipeline/competition/ghost-user", headers=api_headers)
        assert resp.status_code == 404

    def test_returns_competition_data(self, client, api_headers, db_session):
        _seed_competition(db_session, "u-comp-get-1")
        resp = client.get("/pipeline/competition/u-comp-get-1", headers=api_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "competition" in body
        assert body["user_id"] == "u-comp-get-1"

    def test_returned_data_matches_stored(self, client, api_headers, db_session):
        _seed_competition(db_session, "u-comp-get-2")
        resp = client.get("/pipeline/competition/u-comp-get-2", headers=api_headers)
        competition = resp.json()["competition"]
        assert competition["summary"] == "Competitive landscape"
        assert competition["sources_used"] == 12


# ── POST /pipeline/competition/{user_id} ──────────────────────────────────────

class TestPostCompetition:
    def test_missing_idea_returns_425(self, client, api_headers):
        resp = client.post("/pipeline/competition/no-idea-user", headers=api_headers)
        assert resp.status_code == 425

    def test_missing_problems_returns_425(self, client, api_headers, db_session):
        _seed_idea(db_session, "u-comp-no-problems")
        resp = client.post("/pipeline/competition/u-comp-no-problems", headers=api_headers)
        assert resp.status_code == 425

    def test_success_returns_done(self, client, api_headers, db_session):
        _seed_idea(db_session, "u-comp-post-1")
        _seed_problems(db_session, "u-comp-post-1")

        with patch(
            "agents.FiveCompetitionAgent.run_competition_analysis",
            return_value=AGENT_RESULT,
        ):
            resp = client.post("/pipeline/competition/u-comp-post-1", headers=api_headers)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "done"
        assert body["user_id"] == "u-comp-post-1"
        assert "competition" in body


# ── POST /pipeline/competition/{user_id}/chat ──────────────────────────────────

class TestChatCompetition:
    def test_no_competition_data_returns_425(self, client, api_headers):
        payload = {"user_id": "nobody", "message": "who are competitors?", "history": []}
        resp = client.post("/pipeline/competition/nobody/chat", json=payload, headers=api_headers)
        assert resp.status_code == 425

    def test_chat_returns_reply(self, client, api_headers, db_session):
        _seed_competition(db_session, "u-comp-chat-1")

        with patch(
            "agents.FiveCompetitionAgent.chat_competition",
            return_value="Souq is the main competitor.",
        ):
            payload = {"user_id": "u-comp-chat-1", "message": "who is #1?", "history": []}
            resp = client.post(
                "/pipeline/competition/u-comp-chat-1/chat",
                json=payload,
                headers=api_headers,
            )

        assert resp.status_code == 200
        assert resp.json()["reply"] == "Souq is the main competitor."

    def test_chat_increments_history_length(self, client, api_headers, db_session):
        _seed_competition(db_session, "u-comp-chat-2")

        with patch("agents.FiveCompetitionAgent.chat_competition", return_value="Answer"):
            payload = {
                "user_id": "u-comp-chat-2",
                "message": "hello",
                "history": [{"role": "user", "content": "prev"}],
            }
            resp = client.post(
                "/pipeline/competition/u-comp-chat-2/chat",
                json=payload,
                headers=api_headers,
            )

        assert resp.json()["chat_history_length"] == 3
