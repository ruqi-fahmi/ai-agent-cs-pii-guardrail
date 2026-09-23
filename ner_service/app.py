"""
NER Service — model NER dibungkus jadi REST API (ML as a Service).

    uvicorn app:app --app-dir ner_service --port 8020

Kenapa service terpisah dari agent:
  - siklus hidup beda: model bisa di-retrain & dirilis ulang tanpa menyentuh agent
  - profil resource beda: bisa di-scale sendiri sesuai beban
  - bisa dipakai sistem lain (mis. masking di dashboard) lewat kontrak API yang sama

Privasi: service ini menerima teks berisi PII, jadi isi teks TIDAK PERNAH di-log —
hanya panjang teks, jumlah entity, dan latency.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import spacy
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from postprocess import clean_ents, clean_spans

# Dua backend, satu kontrak. Default = model spaCy 40 MB yang dilatih dari nol dan
# menjadi model yang dirilis. NER_BACKEND=indobert memakai fine-tune IndoBERT sebagai
# pembanding (butuh torch + transformers, lihat requirements-indobert.txt).
BACKEND = os.getenv("NER_BACKEND", "spacy").lower()
_default_dir = "indobert" if BACKEND == "indobert" else "model"
MODEL_DIR = Path(os.getenv("NER_MODEL_DIR", Path(__file__).parent / _default_dir))
MAX_CHARS = int(os.getenv("NER_MAX_CHARS", "5000"))

log = logging.getLogger("ner_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
state: dict = {}

# Metrik Prometheus di /metrics — dibaca Managed Prometheus GKE / Grafana.
# Hanya angka agregat; tidak ada label berisi teks (kardinalitas & privasi).
REQUESTS = Counter("ner_requests_total", "Jumlah request /ner yang diproses")
ENTITIES = Counter("ner_entities_total", "Jumlah entity terdeteksi", ["label"])
INFERENCE = Histogram("ner_inference_seconds", "Waktu inferensi model (tanpa HTTP)",
                      buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25))
TEXT_CHARS = Histogram("ner_text_chars", "Panjang teks masukan (karakter)",
                       buckets=(50, 100, 200, 500, 1000, 2000, 5000))


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Muat model SEKALI saat start (~detik), bukan per request.
    # Readiness probe k8s baru hijau setelah ini selesai.
    t0 = time.perf_counter()
    if BACKEND == "indobert":
        from indobert_backend import IndoBertNer
        model = IndoBertNer(MODEL_DIR)
        state["predict"] = lambda teks: clean_spans(teks, model.entities(teks))
        state["version"] = model.version
    else:
        nlp = spacy.load(MODEL_DIR)
        state["predict"] = lambda teks: [(e.start_char, e.end_char, e.label_)
                                         for e in clean_ents(nlp(teks))]
        state["version"] = f"{nlp.meta.get('name', 'ner')}-{nlp.meta.get('version', '0')}"
    state["backend"] = BACKEND
    log.info("backend %s, model %s dimuat dalam %.0f ms", BACKEND, state["version"],
             (time.perf_counter() - t0) * 1000)
    yield
    state.clear()


app = FastAPI(title="PII NER Service", version="1.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class NerRequest(BaseModel):
    text: str = Field(..., max_length=MAX_CHARS,
                      examples=["Nama saya Budi Santoso dan tinggal di Jalan Sudirman Jakarta"])


class Entity(BaseModel):
    label: str   # PERSON | ADDRESS
    text: str
    start: int   # offset karakter, end eksklusif — sama seperti slicing Python
    end: int


class NerResponse(BaseModel):
    entities: list[Entity]
    model_version: str
    latency_ms: float  # waktu inferensi model saja (tanpa overhead HTTP)


@app.get("/health")
def health():
    if "predict" not in state:
        raise HTTPException(status_code=503, detail="model belum dimuat")
    return {"status": "ok", "model_loaded": True, "model_version": state["version"],
            "backend": state.get("backend", "spacy")}


@app.post("/ner", response_model=NerResponse)
def ner(req: NerRequest) -> NerResponse:
    # "def" biasa (bukan async): inferensi spaCy memakan CPU, FastAPI akan
    # menjalankannya di threadpool sehingga event loop tidak ikut tertahan.
    t0 = time.perf_counter()
    spans = state["predict"](req.text)
    elapsed = time.perf_counter() - t0
    latency_ms = elapsed * 1000
    entities = [Entity(label=lab, text=req.text[s:e], start=s, end=e)
                for s, e, lab in spans]
    REQUESTS.inc()
    INFERENCE.observe(elapsed)
    TEXT_CHARS.observe(len(req.text))
    for e in entities:
        ENTITIES.labels(e.label).inc()
    log.info("ner chars=%d entities=%d latency_ms=%.2f", len(req.text), len(entities), latency_ms)
    return NerResponse(entities=entities, model_version=state["version"],
                       latency_ms=round(latency_ms, 3))
