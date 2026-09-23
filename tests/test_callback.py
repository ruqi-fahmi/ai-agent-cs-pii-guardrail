"""Test before_model_callback tanpa Gemini & tanpa NER Service sungguhan (di-mock)."""
import asyncio
from types import SimpleNamespace

import pytest
from google.adk.models import LlmRequest
from google.genai import types

from cs_agent.guardrails import callback, ner_client


def fake_ner(text):
    """Pura-pura NER: tandai 'Budi' dan 'Jalan Melati 5' kalau ada."""
    ents = []
    for value, label in (("Budi", "PERSON"), ("Jalan Melati 5", "ADDRESS")):
        i = text.find(value)
        if i >= 0:
            ents.append({"label": label, "text": value, "start": i, "end": i + len(value)})
    return ents


@pytest.fixture(autouse=True)
def clear_cache():
    callback._cache.clear()


def make_request(*turns):
    return LlmRequest(contents=[
        types.Content(role=role, parts=[types.Part(text=t)]) for role, t in turns])


def run(req):
    ctx = SimpleNamespace(state={})
    return asyncio.run(callback.pii_guardrail(ctx, req)), ctx


def test_redacts_all_user_turns(monkeypatch):
    async def detect(text):
        return fake_ner(text)
    monkeypatch.setattr(ner_client, "detect", detect)

    req = make_request(
        ("user", "Saya Budi, NIK 3201234567890123"),          # giliran LAMA
        ("model", "Baik, ada yang bisa dibantu?"),
        ("user", "Internet di Jalan Melati 5 mati, hubungi 081234567890"),
    )
    resp, ctx = run(req)
    assert resp is None   # lanjut ke Gemini
    texts = [c.parts[0].text for c in req.contents]
    assert texts[0] == "Saya [REDACT_NAMA], NIK [REDACT_NIK]"
    assert texts[1] == "Baik, ada yang bisa dibantu?"          # jawaban model tidak disentuh
    assert texts[2] == "Internet di [REDACT_ADDRESS] mati, hubungi [REDACT_PHONE]"
    assert ctx.state["last_pii_redaction"] == {"PERSON": 1, "NIK": 1, "ADDRESS": 1, "PHONE": 1}


def test_fail_closed_blocks_llm(monkeypatch):
    async def down(text):
        raise ner_client.NerUnavailable("ConnectError")
    monkeypatch.setattr(ner_client, "detect", down)
    monkeypatch.setattr(callback, "FAIL_MODE", "closed")

    resp, _ = run(make_request(("user", "Saya Budi")))
    assert resp is not None                       # Gemini tidak dipanggil
    assert "tidak tersedia" in resp.content.parts[0].text


def test_fail_open_uses_regex_only(monkeypatch):
    async def down(text):
        raise ner_client.NerUnavailable("ConnectError")
    monkeypatch.setattr(ner_client, "detect", down)
    monkeypatch.setattr(callback, "FAIL_MODE", "open")

    req = make_request(("user", "Saya Budi, HP 081234567890"))
    resp, _ = run(req)
    assert resp is None
    assert req.contents[0].parts[0].text == "Saya Budi, HP [REDACT_PHONE]"  # nama lolos


def test_apply_entities_ignores_placeholders():
    text = "hubungi [REDACT_PHONE] ya"
    ents = [{"label": "PERSON", "text": "[REDACT_PHONE]", "start": 8, "end": 22}]
    assert ner_client.apply_entities(text, ents)[0] == text


def run_plugin(text):
    plugin = callback.PiiRedactionPlugin()
    msg = types.Content(role="user", parts=[types.Part(text=text)])
    return asyncio.run(plugin.on_user_message_callback(invocation_context=None, user_message=msg))


def test_plugin_redacts_before_storage(monkeypatch):
    async def detect(text):
        return fake_ner(text)
    monkeypatch.setattr(ner_client, "detect", detect)

    out = run_plugin("Saya Budi, HP 081234567890")
    assert out.parts[0].text == "Saya [REDACT_NAMA], HP [REDACT_PHONE]"


def test_plugin_fail_closed_withholds_raw_text(monkeypatch):
    async def down(text):
        raise ner_client.NerUnavailable("ConnectError")
    monkeypatch.setattr(ner_client, "detect", down)
    monkeypatch.setattr(callback, "FAIL_MODE", "closed")

    out = run_plugin("Saya Budi, HP 081234567890")
    assert out.parts[0].text == callback.WITHHELD      # teks mentah tidak disimpan


def test_already_clean_text_does_not_hit_ner_again(monkeypatch):
    calls = []
    async def detect(text):
        calls.append(text)
        return fake_ner(text)
    monkeypatch.setattr(ner_client, "detect", detect)

    clean, _ = asyncio.run(callback.redact_text("Saya Budi"))
    asyncio.run(callback.redact_text(clean))   # before_model memeriksa ulang teks bersih
    assert len(calls) == 1


@pytest.mark.parametrize("text, expected", [
    ("Halo Bapak/Ibu [REDACT_NAMA], ada yang bisa dibantu?", "Halo Bapak/Ibu, ada yang bisa dibantu?"),
    ("Teknisi akan datang ke alamat [REDACT_ADDRESS].", "Teknisi akan datang ke alamat yang Bapak/Ibu berikan."),
    ("alamat yang Bapak/Ibu berikan ([REDACT_ADDRESS]) sudah tercatat",
     "alamat yang Bapak/Ibu berikan sudah tercatat"),
    ("Kami hubungi ke nomor [REDACT_PHONE]", "Kami hubungi ke nomor yang Bapak/Ibu berikan"),
    ("Tanpa token sama sekali", "Tanpa token sama sekali"),
])
def test_humanize_placeholders(text, expected):
    assert callback.humanize_placeholders(text) == expected


def test_apply_entities_drops_digit_only_person():
    text = "Order ID saya 32012345678901234567 ya"
    ents = [{"label": "PERSON", "text": "32012345678901234567", "start": 14, "end": 34}]
    assert ner_client.apply_entities(text, ents)[0] == text


def test_analyze_spans_point_to_original_text(monkeypatch):
    """Posisi dari NER (dihitung di teks antara yang berisi placeholder) harus
    dipetakan balik tepat ke teks ASLI — halaman demo mewarnai berdasarkan ini."""
    async def detect(text):
        return fake_ner(text)
    monkeypatch.setattr(ner_client, "detect", detect)

    text = "HP 081234567890, saya Budi, di Jalan Melati 5 ya"
    a = asyncio.run(callback.analyze_text(text))
    got = [(text[s["start"]:s["end"]], s["type"], s["layer"]) for s in a.spans]
    assert got == [("081234567890", "PHONE", "regex"),
                   ("Budi", "PERSON", "ner"),
                   ("Jalan Melati 5", "ADDRESS", "ner")]
    assert a.clean == "HP [REDACT_PHONE], saya [REDACT_NAMA], di [REDACT_ADDRESS] ya"


def test_fallback_gemini_switches_model_on_quota(monkeypatch):
    from google.adk.models import LlmResponse as Resp
    from google.adk.models.google_llm import Gemini
    from cs_agent.model import FallbackGemini

    calls = []
    async def fake_generate(self, llm_request, stream=False):
        calls.append(llm_request.model)
        if llm_request.model == "utama":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        yield Resp(content=types.Content(role="model", parts=[types.Part(text="ok")]))
    monkeypatch.setattr(Gemini, "generate_content_async", fake_generate)

    model = FallbackGemini(model="utama", fallbacks=["cadangan"])
    async def collect():
        return [r async for r in model.generate_content_async(LlmRequest(model="utama"))]
    out = asyncio.run(collect())
    assert calls == ["utama", "cadangan"]
    assert out[0].custom_metadata == {"served_by": "cadangan", "fallback_from": ["utama"]}


def test_fallback_gemini_does_not_retry_other_errors(monkeypatch):
    from google.adk.models.google_llm import Gemini
    from cs_agent.model import FallbackGemini

    async def boom(self, llm_request, stream=False):
        raise RuntimeError("400 INVALID_ARGUMENT")
        yield  # pragma: no cover
    monkeypatch.setattr(Gemini, "generate_content_async", boom)
    model = FallbackGemini(model="utama", fallbacks=["cadangan"])
    async def collect():
        return [r async for r in model.generate_content_async(LlmRequest(model="utama"))]
    with pytest.raises(RuntimeError, match="400"):
        asyncio.run(collect())


def test_currency_symbol_is_not_a_person():
    """Ditemukan test browser: model menebak "Rp" (dari "Rp1508000") sebagai PERSON."""
    text = "Tagihan saya Rp1508000 kok naik ya?"
    ents = [{"label": "PERSON", "text": "Rp", "start": 13, "end": 15}]
    assert ner_client.apply_entities(text, ents)[0] == text
