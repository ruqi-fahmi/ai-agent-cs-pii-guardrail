"""
Guardrail PII lapis 2: panggil NER Service lewat HTTP, redaksi nama & alamat.
"""
import os

import httpx

NER_SERVICE_URL = os.getenv("NER_SERVICE_URL", "http://127.0.0.1:8020")
NER_TIMEOUT_S = float(os.getenv("NER_TIMEOUT_S", "2.0"))

PLACEHOLDER = {
    "PERSON": "[REDACT_NAMA]",
    "ADDRESS": "[REDACT_ADDRESS]",
}

# Satu client dipakai ulang -> koneksi TCP tetap hidup (keep-alive), tidak
# membuka koneksi baru di setiap pesan.
_client = httpx.AsyncClient(base_url=NER_SERVICE_URL, timeout=NER_TIMEOUT_S)


class NerUnavailable(Exception):
    """NER Service tidak bisa dihubungi / error. Callback yang memutuskan nasibnya."""


async def detect(text: str) -> list[dict]:
    try:
        resp = await _client.post("/ner", json={"text": text})
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise NerUnavailable(f"{type(exc).__name__}: {exc}") from exc
    return resp.json()["entities"]


def plausible_entities(text: str, entities: list[dict]) -> list[dict]:
    """Aturan kewarasan di atas tebakan model:
      - tebakan yang menyentuh placeholder regex diabaikan ("[REDACT_PHONE]" bukan nama)
      - PERSON tanpa satu huruf pun dibuang: nama orang pasti mengandung huruf.
        Contoh nyata: model menandai order ID "32012345678901234567" sebagai PERSON.
    """
    def ok(e: dict) -> bool:
        span = text[e["start"]:e["end"]]
        if e["label"] not in PLACEHOLDER or "[" in span or "]" in span:
            return False
        if e["label"] == "PERSON":
            if not any(ch.isalpha() for ch in span):
                return False
            # Simbol mata uang bukan nama. Contoh nyata (ditemukan test browser):
            # "Rp1508000" dipecah tokenizer jadi "Rp" + angka, lalu "Rp" ditebak PERSON.
            if span.strip().lower() in CURRENCY:
                return False
        return True
    return [e for e in entities if ok(e)]


CURRENCY = {"rp", "rp.", "idr", "usd", "us$", "sgd", "myr"}


def apply_entities(text: str, entities: list[dict]) -> tuple[str, list[str]]:
    """Ganti span entity dengan placeholder. Mengembalikan (teks, daftar label)."""
    spans = plausible_entities(text, entities)
    # Ganti dari BELAKANG ke depan: kalau dari depan, panjang teks berubah dan
    # offset entity berikutnya jadi meleset.
    for e in sorted(spans, key=lambda e: e["start"], reverse=True):
        text = text[:e["start"]] + PLACEHOLDER[e["label"]] + text[e["end"]:]
    return text, [e["label"] for e in spans]
