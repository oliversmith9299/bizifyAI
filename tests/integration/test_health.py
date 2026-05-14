"""
Integration tests for the health check endpoint.
"""


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_returns_running_status(self, client):
        resp = client.get("/")
        body = resp.json()
        assert body["status"] == "AI service running"
