from fastapi.testclient import TestClient

from server import app


def test_status_and_index():
    with TestClient(app) as client:
        s = client.get("/api/status").json()
        assert {"ner", "gemini_model", "gemini_fallbacks", "fail_mode"} <= s.keys()
        assert "PII Guardrail" in client.get("/").text
