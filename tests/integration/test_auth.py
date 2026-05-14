"""
Integration tests for API key authentication.
Verifies that the verify_api_key dependency correctly gates all protected routes.
"""

import pytest


class TestApiKeyAuth:
    def test_missing_api_key_returns_422(self, client):
        # FastAPI returns 422 Unprocessable Entity when a required Header is absent
        resp = client.get("/pipeline/customers/some-user")
        assert resp.status_code == 422

    def test_wrong_api_key_returns_401(self, client):
        resp = client.get(
            "/pipeline/customers/some-user",
            headers={"X-API-KEY": "completely-wrong-key"},
        )
        assert resp.status_code == 401

    def test_correct_api_key_is_accepted(self, client, api_headers):
        # 404 means auth passed but user not found — auth itself worked
        resp = client.get("/pipeline/customers/nonexistent-user", headers=api_headers)
        assert resp.status_code == 404

    def test_wrong_key_detail_message(self, client):
        resp = client.get(
            "/pipeline/customers/user",
            headers={"X-API-KEY": "bad-key"},
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]
