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
# Model directory defaults
SPACY_DIR = Path(os.getenv("NER_MODEL_DIR", Path(__file__).parent / "model"))
INDOBERT_DIR = Path(os.getenv("NER_INDOBERT_DIR",
                              Path(__file__).parent.parent / ".cache" / "indobert" / "model"))
if not INDOBERT_DIR.exists() and (Path(__file__).parent / "indobert").exists():
    INDOBERT_DIR = Path(__file__).parent / "indobert"

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
    # Muat model saat start (~detik), bukan per request.
    t0 = time.perf_counter()
    # 1. Model spaCy (bawaan)
    nlp = spacy.load(SPACY_DIR)
    state["spacy"] = {
        "predict": lambda teks: [(e.start_char, e.end_char, e.label_) for e in clean_ents(nlp(teks))],
        "version": f"{nlp.meta.get('name', 'ner')}-{nlp.meta.get('version', '0')}",
        "name": "spaCy (Model Kita)"
    }
    log.info("backend spacy (%s) dimuat dalam %.0f ms", state["spacy"]["version"],
             (time.perf_counter() - t0) * 1000)

    # 2. Model IndoBERT (opsional, jika tersedia di disk)
    t1 = time.perf_counter()
    if INDOBERT_DIR.exists():
        try:
            from indobert_backend import IndoBertNer
            indobert = IndoBertNer(INDOBERT_DIR)
            state["indobert"] = {
                "predict": lambda teks: clean_spans(teks, indobert.entities(teks)),
                "version": indobert.version,
                "name": "IndoBERT (Transformer)"
            }
            log.info("backend indobert (%s) dimuat dalam %.0f ms", indobert.version,
                     (time.perf_counter() - t1) * 1000)
        except Exception as exc:
            log.warning("Gagal memuat IndoBERT (%s) -> hanya spaCy aktif", exc)
            state["indobert"] = None
    else:
        state["indobert"] = None

    state["active"] = os.getenv("NER_BACKEND", "spacy").lower()
    yield
    state.clear()


app = FastAPI(title="PII NER Service", version="1.2.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class NerRequest(BaseModel):
    text: str = Field(..., max_length=MAX_CHARS,
                      examples=["Nama saya Budi Santoso dan tinggal di Jalan Sudirman Jakarta"])
    backend: str | None = None  # "spacy" | "indobert" (default ke active)


class Entity(BaseModel):
    label: str   # PERSON | ADDRESS
    text: str
    start: int   # offset karakter, end eksklusif — sama seperti slicing Python
    end: int


class NerResponse(BaseModel):
    entities: list[Entity]
    model_version: str
    latency_ms: float  # waktu inferensi model saja (tanpa overhead HTTP)
    backend: str = "spacy"


@app.get("/health")
def health():
    if "spacy" not in state:
        raise HTTPException(status_code=503, detail="model belum dimuat")
    active = state.get("active", "spacy")
    active_entry = state.get(active) or state["spacy"]
    backends = {
        "spacy": {"available": "spacy" in state, "version": state["spacy"]["version"]},
        "indobert": {"available": state.get("indobert") is not None,
                     "version": state["indobert"]["version"] if state.get("indobert") else None}
    }
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": active_entry["version"],
        "backend": active,
        "backends": backends
    }


BACKENDS = ("spacy", "indobert")     # daftar tertutup: "state" juga memuat kunci lain


@app.post("/backend/{name}")
def switch_backend(name: str):
    name = name.lower()
    # Tanpa daftar tertutup ini, nama seperti "active" lolos (kebetulan ada di state),
    # lalu state["active"] terisi nilai yang bukan backend dan /ner ikut gagal.
    if name not in BACKENDS:
        raise HTTPException(status_code=404, detail=f"backend '{name}' tidak dikenal")
    if state.get(name) is None:
        raise HTTPException(status_code=400, detail=f"backend '{name}' tidak tersedia")
    state["active"] = name
    return {"status": "ok", "active": name, "version": state[name]["version"]}


@app.post("/ner", response_model=NerResponse)
def ner(req: NerRequest) -> NerResponse:
    target = (req.backend or state.get("active", "spacy")).lower()
    backend_data = state.get(target) if target in BACKENDS else None
    if not backend_data:                  # diminta backend yang tidak ada -> pakai bawaan
        backend_data = state["spacy"]
        target = "spacy"

    t0 = time.perf_counter()
    spans = backend_data["predict"](req.text)
    elapsed = time.perf_counter() - t0
    latency_ms = elapsed * 1000
    entities = [Entity(label=lab, text=req.text[s:e], start=s, end=e)
                for s, e, lab in spans]
    REQUESTS.inc()
    INFERENCE.observe(elapsed)
    TEXT_CHARS.observe(len(req.text))
    for e in entities:
        ENTITIES.labels(e.label).inc()
    log.info("ner backend=%s chars=%d entities=%d latency_ms=%.2f",
             target, len(req.text), len(entities), latency_ms)
    return NerResponse(entities=entities, model_version=backend_data["version"],
                       latency_ms=round(latency_ms, 3), backend=target)
