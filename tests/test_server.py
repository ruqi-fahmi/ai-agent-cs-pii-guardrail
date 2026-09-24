import pytest
from fastapi.testclient import TestClient

from cs_agent.guardrails import callback, ner_client
from server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_status_and_index():
    with TestClient(app) as client:
        s = client.get("/api/status").json()
        assert {"ner", "gemini_model", "gemini_fallbacks", "fail_mode"} <= s.keys()
        assert "PII Guardrail" in client.get("/").text


def test_tukar_backend_ditolak_kalau_alt_tidak_diatur(client, monkeypatch):
    """Tombol 'tukar model' hanya berarti kalau server punya NER Service kedua."""
    monkeypatch.setattr(ner_client, "NER_SERVICE_URL_ALT", "")
    r = client.post("/api/backend", json={"which": "alt"})
    assert r.status_code == 409


def test_tukar_backend_pindah_url_dan_membuang_cache(client, monkeypatch):
    monkeypatch.setattr(ner_client, "NER_SERVICE_URL_ALT", "http://127.0.0.1:8099")
    callback._cache["sisa"] = "dari model lama"
    r = client.post("/api/backend", json={"which": "alt"})
    assert r.status_code == 200
    assert r.json()["ner_url"] == "http://127.0.0.1:8099"
    assert not callback._cache            # hasil model lama tidak boleh dipakai ulang
    client.post("/api/backend", json={"which": "utama"})   # kembalikan
    assert ner_client.active_url() == ner_client.NER_SERVICE_URL
