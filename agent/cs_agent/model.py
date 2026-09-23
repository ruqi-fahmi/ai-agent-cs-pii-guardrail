"""
Gemini dengan fallback antar-model saat kuota habis / server sibuk.

Kuota free tier Gemini dihitung PER MODEL (gemini-2.5-flash: 20 request/hari saat
diuji 22 Sep 2026). Tanpa fallback, demo mati begitu kuota satu model habis.
Pola yang sama dengan insights.py di project Bola Gembira: coba model utama,
kalau 429/503 pindah ke model berikutnya.
"""
import logging
import os
import re
from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.models.google_llm import Gemini
from google.genai import types

log = logging.getLogger("cs_agent.model")

PRIMARY = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACKS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACK_MODELS",
    "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-flash-latest").split(",") if m.strip()]


def _retryable(exc: Exception) -> bool:
    """429 kuota habis, 503 server sibuk, 404 model sudah ditutup (mis.
    gemini-2.5-flash-lite: "no longer available to new users" padahal masih
    muncul di daftar model) -> layak dicoba dengan model berikutnya."""
    text = f"{type(exc).__name__} {exc}"
    return any(k in text for k in ("429", "RESOURCE_EXHAUSTED", "ResourceExhausted",
                                   "503", "UNAVAILABLE", "overloaded",
                                   "404", "NOT_FOUND"))


class FallbackGemini(Gemini):
    fallbacks: list[str] = []

    async def generate_content_async(self, llm_request: LlmRequest,
                                     stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        models = [llm_request.model or self.model, *[m for m in self.fallbacks if m != self.model]]
        tried = []
        for i, model in enumerate(models):
            llm_request.model = model
            try:
                async for resp in super().generate_content_async(llm_request, stream):
                    # Catat model yang benar-benar menjawab + yang gagal sebelumnya,
                    # dibaca after_model_callback untuk halaman demo.
                    resp.model_version = resp.model_version or model
                    resp.custom_metadata = {**(resp.custom_metadata or {}),
                                            "served_by": model, "fallback_from": tried}
                    yield resp
                return
            except Exception as exc:  # noqa: BLE001 — disaring oleh _retryable
                if i == len(models) - 1 or not _retryable(exc):
                    raise
                log.warning("model %s gagal (%s) -> coba %s", model,
                            type(exc).__name__, models[i + 1])
                tried.append(model)


class FakeLlm(BaseLlm):
    """LLM 'boneka' untuk test end-to-end (CS_AGENT_FAKE_LLM=1).

    Semua guardrail tetap berjalan sungguhan (plugin, before/after_model_callback,
    NER Service) — hanya panggilan ke Gemini yang diganti, supaya test gratis,
    cepat, konsisten, dan bisa jalan di CI tanpa API key. Jawabannya sengaja
    menyebut ulang token [REDACT_*] yang diterima, sehingga test bisa memeriksa
    (1) apa yang sampai ke LLM dan (2) bahwa after_model_callback merapikannya.
    """
    model: str = "fake-llm"

    async def generate_content_async(self, llm_request: LlmRequest,
                                     stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        last_user = next((c for c in reversed(llm_request.contents) if c.role == "user"), None)
        text = "".join(p.text or "" for p in (last_user.parts or [])) if last_user else ""
        tokens = sorted(set(re.findall(r"\[REDACT_[A-Z]+\]", text)))
        reply = ("Baik, data " + ", ".join(tokens) + " sudah kami terima." if tokens
                 else "Baik, ada yang bisa kami bantu?")
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=reply)]),
                          model_version=self.model)


def build_model() -> BaseLlm:
    if os.getenv("CS_AGENT_FAKE_LLM") == "1":
        log.warning("CS_AGENT_FAKE_LLM=1 -> memakai LLM palsu (khusus test)")
        return FakeLlm()
    return FallbackGemini(model=PRIMARY, fallbacks=FALLBACKS)
