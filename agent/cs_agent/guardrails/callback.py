"""
Tiga titik jaga PII di siklus hidup agent:

  1. PiiRedactionPlugin.on_user_message_callback  — pintu masuk SESI
     Pesan user disensor SEBELUM disimpan ke riwayat sesi, jadi memori server pun
     tidak pernah memegang PII mentah.
  2. pii_guardrail (before_model_callback)        — pintu masuk LLM (syarat soal)
     Lapis kedua (defense in depth): semua teks user di llm_request diperiksa lagi
     tepat sebelum dikirim ke Gemini. Menangkap jalur yang tidak lewat plugin
     (mis. pesan yang disuntikkan langsung ke sesi, runner tanpa plugin).
  3. placeholder_cleanup (after_model_callback)   — pintu KELUAR
     Token [REDACT_*] yang dibeo LLM diganti frasa wajar.

Kenapa guardrail tidak dijadikan tool: tool dipanggil atas kehendak LLM, artinya
LLM sudah membaca pesannya duluan.

Setiap titik juga mencatat JEJAK (TRACES) per sesi — posisi PII, lapis yang
menangkap, waktu, dan payload persis ke Gemini — untuk halaman demo. Jejak tidak
pernah memuat nilai PII, hanya posisi karakter.
"""
import hashlib
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from . import ner_client, regex_pii

log = logging.getLogger("cs_agent.guardrail")

# closed = NER mati -> tidak ada teks mentah yang disimpan maupun dikirim ke Gemini.
#          Default, karena untuk guardrail PII, bocor lebih mahal daripada gagal menjawab.
# open   = NER mati -> tetap lanjut dengan regex saja (nama/alamat bisa lolos).
FAIL_MODE = os.getenv("GUARDRAIL_FAIL_MODE", "closed").lower()
DEBUG = os.getenv("GUARDRAIL_DEBUG", "0") == "1"

REFUSAL = ("Mohon maaf, sistem pengaman data kami sedang tidak tersedia sehingga pesan "
           "Anda belum bisa diproses. Silakan coba beberapa saat lagi.")
# Pengganti pesan yang tidak bisa disensor saat fail-closed (disimpan di sesi).
WITHHELD = "[PESAN DITAHAN: pengaman data tidak tersedia]"

PLACEHOLDER = {**regex_pii.PLACEHOLDER, **ner_client.PLACEHOLDER}


@dataclass
class Analysis:
    clean: str
    spans: list[dict] = field(default_factory=list)   # {start, end, type, layer} di teks ASLI
    counts: dict = field(default_factory=dict)
    regex_ms: float = 0.0
    ner_ms: float = 0.0


# Hasil analisis di-cache per teks (kunci = hash, jadi cache tidak memegang teks asli).
# Teks yang SUDAH bersih juga dicatat, sehingga saat before_model_callback memeriksa
# ulang riwayat, pesan lama tidak dikirim ke NER lagi.
_cache: dict[str, Analysis] = {}
_CACHE_MAX = 1000


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _count(items) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


async def analyze_text(text: str) -> Analysis:
    """Regex lalu NER. Posisi setiap PII dikembalikan dalam koordinat teks ASLI.

    NerUnavailable dibiarkan naik — pemanggil yang memutuskan fail-closed/open.
    """
    if (hit := _cache.get(_key(text))) is not None:
        return hit

    # Lapis 1 — regex, dijalankan DULUAN supaya NIK/email/telepon sudah hilang
    # sebelum teks dikirim lewat jaringan ke NER Service.
    t0 = time.perf_counter()
    findings = regex_pii.find(text)
    regex_ms = (time.perf_counter() - t0) * 1000

    # Teks antara = asli dengan temuan regex diganti placeholder. Catat segmennya
    # supaya posisi tebakan NER (di teks antara) bisa dipetakan balik ke teks asli.
    inter, segments, pos = "", [], 0      # segment: (awal_antara, akhir_antara, awal_asli, placeholder?)
    for f in findings:
        chunk = text[pos:f.start]
        segments.append((len(inter), len(inter) + len(chunk), pos, False))
        inter += chunk
        ph = regex_pii.PLACEHOLDER[f.type]
        segments.append((len(inter), len(inter) + len(ph), f.start, True))
        inter += ph
        pos = f.end
    segments.append((len(inter), len(inter) + len(text) - pos, pos, False))
    inter += text[pos:]

    # Lapis 2 — NER (nama & alamat)
    t1 = time.perf_counter()
    entities = ner_client.plausible_entities(inter, await ner_client.detect(inter))
    ner_ms = (time.perf_counter() - t1) * 1000

    spans = [{"start": f.start, "end": f.end, "type": f.type, "layer": "regex"} for f in findings]
    for e in entities:
        for s_start, s_end, o_start, is_ph in segments:
            if not is_ph and s_start <= e["start"] and e["end"] <= s_end:
                off = o_start - s_start
                spans.append({"start": e["start"] + off, "end": e["end"] + off,
                              "type": e["label"], "layer": "ner"})
                break
    spans.sort(key=lambda s: s["start"])

    clean = regex_pii.replace_spans(text, [(s["start"], s["end"], s["type"]) for s in spans],
                                    PLACEHOLDER)
    result = Analysis(clean, spans, _count(s["type"] for s in spans), regex_ms, ner_ms)
    if len(_cache) >= _CACHE_MAX:
        _cache.clear()
    _cache[_key(text)] = result
    _cache[_key(clean)] = Analysis(clean, [], result.counts)
    return result


def clear_cache() -> None:
    """Buang hasil redaksi yang tersimpan. Wajib dipanggil saat model NER ditukar,
    supaya jawaban model lama tidak dipakai ulang untuk teks yang sama."""
    _cache.clear()


async def redact_text(text: str) -> tuple[str, dict]:
    """Bentuk ringkas: (teks_bersih, hitungan per jenis PII)."""
    a = await analyze_text(text)
    return a.clean, a.counts


def _regex_only(text: str) -> Analysis:
    t0 = time.perf_counter()
    findings = regex_pii.find(text)
    spans = [{"start": f.start, "end": f.end, "type": f.type, "layer": "regex"} for f in findings]
    clean = regex_pii.replace_spans(text, [(f.start, f.end, f.type) for f in findings], PLACEHOLDER)
    return Analysis(clean, spans, _count(f.type for f in findings),
                    (time.perf_counter() - t0) * 1000)


# --- Jejak untuk halaman demo ---------------------------------------------------
# Per sesi, hanya giliran terakhir. Isinya posisi & angka, BUKAN nilai PII —
# kecuali payload ke Gemini, yang memang sudah bersih.
TRACES: "OrderedDict[str, dict]" = OrderedDict()
_TRACES_MAX = 500


def _trace(session_id: Optional[str]) -> dict:
    if not session_id:
        return {}
    if session_id not in TRACES:
        TRACES[session_id] = {}
        while len(TRACES) > _TRACES_MAX:
            TRACES.popitem(last=False)
    return TRACES[session_id]


def _session_id(ctx) -> Optional[str]:
    session = getattr(ctx, "session", None)
    return getattr(session, "id", None)


# --- 1. Plugin: pintu masuk sesi ------------------------------------------------
class PiiRedactionPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="pii_redaction")

    async def on_user_message_callback(self, *, invocation_context,
                                       user_message: types.Content) -> Optional[types.Content]:
        parts, total = [], {}
        trace = _trace(_session_id(invocation_context))
        trace.clear()
        for part in user_message.parts or []:
            if not part.text:
                parts.append(part)
                continue
            withheld = False
            try:
                a = await analyze_text(part.text)
            except ner_client.NerUnavailable as exc:
                if FAIL_MODE == "closed":
                    # Jangan simpan teks mentah. before_model_callback nanti
                    # memblokir panggilan ke Gemini karena NER masih mati.
                    log.error("NER tidak tersedia (%s) -> pesan ditahan", exc)
                    a, withheld = Analysis(WITHHELD), True
                else:
                    log.warning("NER tidak tersedia (%s) -> fail-open, hanya regex", exc)
                    a = _regex_only(part.text)
            parts.append(types.Part(text=a.clean))
            for k, v in a.counts.items():
                total[k] = total.get(k, 0) + v
            if "spans" not in trace:          # halaman demo menampilkan bagian teks pertama
                trace.update(spans=a.spans, regex_ms=round(a.regex_ms, 2),
                             ner_ms=round(a.ner_ms, 2), withheld=withheld,
                             stored_text=a.clean,
                             ner_backend=ner_client.active_backend())
        log.info("plugin: pesan disimpan ke sesi setelah redaksi %s", total or "tidak ada PII")
        # Mengembalikan Content = ADK MENGGANTI pesan user dengan versi ini,
        # termasuk yang disimpan di riwayat sesi.
        return types.Content(role=user_message.role, parts=parts)


def _instruction_text(llm_request: LlmRequest) -> str:
    si = llm_request.config.system_instruction if llm_request.config else None
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    parts = getattr(si, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)


# --- 2. before_model_callback: pintu masuk LLM ----------------------------------
async def pii_guardrail(callback_context: CallbackContext,
                        llm_request: LlmRequest) -> Optional[LlmResponse]:
    trace = _trace(_session_id(callback_context))
    total: dict[str, int] = {}
    for content in llm_request.contents:
        if content.role != "user":
            continue
        for part in content.parts or []:
            if not part.text:   # mis. hasil tool (function_response) — bukan teks user
                continue
            try:
                a = await analyze_text(part.text)
            except ner_client.NerUnavailable as exc:
                if FAIL_MODE == "closed":
                    log.error("NER tidak tersedia (%s) -> fail-closed, Gemini tidak dipanggil", exc)
                    trace.update(blocked=True, llm_contents=[])
                    # Mengembalikan LlmResponse = ADK MELEWATI panggilan ke model
                    # dan memakai respons ini sebagai jawaban.
                    return LlmResponse(content=types.Content(
                        role="model", parts=[types.Part(text=REFUSAL)]))
                log.warning("NER tidak tersedia (%s) -> fail-open, hanya regex", exc)
                a = _regex_only(part.text)
            part.text = a.clean
            for k, v in a.counts.items():
                total[k] = total.get(k, 0) + v

    # Yang di-log hanya JUMLAH per jenis, tidak pernah isi PII-nya.
    log.info("guardrail: redaksi %s", total or "tidak ada PII")
    if DEBUG:
        # Untuk demo saja: tampilkan teks yang BENAR-BENAR dikirim ke Gemini.
        # Mati secara default — kalau NER melewatkan sebuah nama (FN), log ini
        # ikut membocorkannya.
        for content in llm_request.contents:
            if content.role == "user":
                for part in content.parts or []:
                    if part.text:
                        log.info("ke Gemini >> %s", part.text)
    # Jejak: payload PERSIS yang akan dikirim — diambil dari llm_request itu sendiri.
    trace.update(
        blocked=False,
        model=llm_request.model,
        system_instruction=_instruction_text(llm_request),
        llm_contents=[{"role": c.role, "text": "".join(p.text or "" for p in (c.parts or []))}
                      for c in llm_request.contents],
        _llm_t0=time.perf_counter())
    callback_context.state["last_pii_redaction"] = total
    return None   # None = lanjutkan ke Gemini dengan llm_request yang sudah bersih


# --- 3. after_model_callback: pintu KELUAR --------------------------------------
# Walau sudah dilarang di instruksi, LLM kadang membeo token "[REDACT_NAMA]" ke
# pengguna. Di sini token diganti frasa wajar. Nilai asli TIDAK dikembalikan —
# agent memang tidak pernah tahu nilainya.
_FRIENDLY = [
    # "(alamat yang Bapak/Ibu berikan)" setelah frasa yang sama -> buang duplikatnya
    (re.compile(r"\s*\(\[REDACT_[A-Z]+\]\)"), ""),
    (re.compile(r"\b(Bapak/Ibu|Bapak|Ibu|Kak|kak)\s+\[REDACT_NAMA\]"), r"\1"),
    (re.compile(r"\[REDACT_NAMA\]"), "Bapak/Ibu"),
    (re.compile(r"\balamat\s+\[REDACT_ADDRESS\]", re.I), "alamat yang Bapak/Ibu berikan"),
    (re.compile(r"\[REDACT_ADDRESS\]"), "alamat yang Bapak/Ibu berikan"),
    (re.compile(r"\b(nomor(?: telepon| HP)?)\s+\[REDACT_PHONE\]", re.I), r"\1 yang Bapak/Ibu berikan"),
    (re.compile(r"\[REDACT_PHONE\]"), "nomor telepon yang Bapak/Ibu berikan"),
    (re.compile(r"\b(email|e-mail|alamat email)\s+\[REDACT_EMAIL\]", re.I), r"\1 yang Bapak/Ibu berikan"),
    (re.compile(r"\[REDACT_EMAIL\]"), "email yang Bapak/Ibu berikan"),
    (re.compile(r"\b(NIK)\s+\[REDACT_NIK\]"), r"\1 yang Bapak/Ibu berikan"),
    (re.compile(r"\[REDACT_NIK\]"), "NIK yang Bapak/Ibu berikan"),
]


def humanize_placeholders(text: str) -> str:
    for pattern, repl in _FRIENDLY:
        text = pattern.sub(repl, text)
    return text


def placeholder_cleanup(callback_context: CallbackContext,
                        llm_response: LlmResponse) -> Optional[LlmResponse]:
    trace = _trace(_session_id(callback_context))
    if "_llm_t0" in trace:
        trace["llm_ms"] = round((time.perf_counter() - trace.pop("_llm_t0")) * 1000, 1)
    meta = llm_response.custom_metadata or {}
    trace["served_by"] = meta.get("served_by") or llm_response.model_version
    trace["fallback_from"] = meta.get("fallback_from", [])
    if not llm_response.content or not llm_response.content.parts:
        return None
    changed = False
    for part in llm_response.content.parts:
        if part.text and "[REDACT_" in part.text:
            trace["raw_reply_had_placeholders"] = True
            part.text = humanize_placeholders(part.text)
            changed = True
    return llm_response if changed else None
