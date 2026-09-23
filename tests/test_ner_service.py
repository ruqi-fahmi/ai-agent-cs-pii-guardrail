from fastapi.testclient import TestClient

from app import app


def test_health_and_ner_contract():
    with TestClient(app) as client:   # "with" -> lifespan jalan -> model dimuat
        assert client.get("/health").json()["model_loaded"] is True

        text = "Nama saya Budi Santoso dan tinggal di Jalan Sudirman Jakarta"
        body = client.post("/ner", json={"text": text}).json()
        assert {"entities", "model_version", "latency_ms"} <= body.keys()
        found = {(e["label"], e["text"]) for e in body["entities"]}
        assert ("PERSON", "Budi Santoso") in found
        assert ("ADDRESS", "Jalan Sudirman Jakarta") in found
        # offset konsisten dengan teks
        for e in body["entities"]:
            assert text[e["start"]:e["end"]] == e["text"]


def test_rejects_invalid_input():
    with TestClient(app) as client:
        assert client.post("/ner", json={}).status_code == 422
        assert client.post("/ner", json={"text": "a" * 5001}).status_code == 422


def test_metrics_endpoint_counts_requests():
    with TestClient(app) as client:
        client.post("/ner", json={"text": "Nama saya Budi Santoso"})
        body = client.get("/metrics/").text
        assert "ner_requests_total" in body
        assert "ner_inference_seconds_bucket" in body
        assert "Budi" not in body          # metrik tidak boleh memuat isi teks


def test_clean_ents_trims_common_words_from_person():
    import spacy
    from postprocess import clean_ents

    nlp = spacy.blank("id")
    text = "apakah status tagihan untuk pelanggan agustinus lamere di tebet dalam jakarta"
    doc = nlp(text)

    def span(frag, label):
        i = text.index(frag)
        return doc.char_span(i, i + len(frag), label)

    doc.ents = [span("apakah status", "PERSON"),
                span("untuk pelanggan agustinus lamere", "PERSON"),
                span("tebet dalam jakarta", "ADDRESS")]
    got = [(e.label_, e.text) for e in clean_ents(doc)]
    assert got == [("PERSON", "agustinus lamere"), ("ADDRESS", "tebet dalam jakarta")]


def test_no_known_name_is_a_common_word():
    """Daftar kata umum tidak boleh memangkas nama asli (generator, val, dan semua test)."""
    import sys
    from pathlib import Path

    from ner_data import load_jsonl
    from postprocess import COMMON

    data = Path(__file__).parents[1] / "ner_service" / "data"
    sys.path.insert(0, str(data))
    import generate_train as g

    tokens = {n.lower() for n in g.FIRST_NAMES + g.LAST_NAMES}
    for f in data.glob("*.jsonl"):
        for r in load_jsonl(f):
            for s, e, label in r["entities"]:
                if label == "PERSON":
                    tokens |= set(r["text"][s:e].lower().split())
    assert not tokens & COMMON, sorted(tokens & COMMON)
