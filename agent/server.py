"""
Server demo: halaman chat + panel bukti guardrail.

    uvicorn server:app --app-dir agent --port 8021      # buka http://127.0.0.1:8021

Menjalankan agent yang SAMA dengan `adk web` (App + plugin + callback), tapi selain
jawaban juga mengembalikan JEJAK guardrail tiap giliran: posisi PII di pesan asli,
lapis yang menangkap (regex/NER), payload persis yang dikirim ke Gemini (diambil
dari before_model_callback), dan waktu tiap tahap.

Privasi: jejak hanya memuat POSISI karakter, bukan nilai PII. Browser mewarnai
pesan asli sendiri — teksnya memang sudah ada di browser karena pengguna yang mengetik.
"""
import copy
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel, Field

from cs_agent.agent import app as adk_app
from cs_agent.guardrails import callback, ner_client

WEB = Path(__file__).parent / "web"
USER_ID = "demo"

runner = InMemoryRunner(app=adk_app)
app = FastAPI(title="CS Agent — PII Guardrail Demo")


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=4000)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
async def status():
    ner = {"ok": False, "model_version": None}
    try:
        r = await ner_client._client.get("/health")
        if r.status_code == 200:
            ner = {"ok": True, "model_version": r.json().get("model_version")}
    except Exception:  # noqa: BLE001 — status saja, bukan jalur kritis
        pass
    model = adk_app.root_agent.model
    return {"ner": ner, "ner_url": ner_client.NER_SERVICE_URL,
            "gemini_model": getattr(model, "model", model),
            "gemini_fallbacks": getattr(model, "fallbacks", []),
            "fail_mode": callback.FAIL_MODE}


@app.post("/api/session")
async def new_session():
    s = await runner.session_service.create_session(app_name=adk_app.name, user_id=USER_ID)
    return {"session_id": s.id}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    t0 = time.perf_counter()
    reply, error = [], None
    try:
        async for event in runner.run_async(
                user_id=USER_ID, session_id=req.session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=req.message)])):
            if event.is_final_response() and event.content and event.content.parts:
                reply.append("".join(p.text or "" for p in event.content.parts))
    except ValueError as exc:            # sesi tidak dikenal (mis. server di-restart)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # Gemini gagal SETELAH guardrail berjalan: tetap kirim jejaknya, karena
        # sensor & payload-nya sudah terjadi dan itu yang ingin dibuktikan.
        text = f"{type(exc).__name__} {exc}"
        error = ("Kuota Gemini habis untuk semua model (free tier). Guardrail tetap "
                 "berjalan — lihat panel bukti. Coba lagi nanti atau ganti API key."
                 if "429" in text or "RESOURCE_EXHAUSTED" in text
                 else f"Gemini gagal: {type(exc).__name__}")

    trace = {k: v for k, v in copy.deepcopy(callback.TRACES.get(req.session_id, {})).items()
             if not k.startswith("_")}
    trace["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return {"reply": "\n".join(reply).strip() or None, "error": error, "trace": trace}
