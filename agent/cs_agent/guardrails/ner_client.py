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

# Alamat NER Service kedua (opsional). Dipakai halaman demo untuk memperlihatkan bahwa
# model bisa ditukar tanpa menyentuh agent — lihat docs/ner-iterasi.md, dua backend.
NER_SERVICE_URL_ALT = os.getenv("NER_SERVICE_URL_ALT", "")

# Satu client dipakai ulang -> koneksi TCP tetap hidup (keep-alive), tidak
# membuka koneksi baru di setiap pesan.
_client = httpx.AsyncClient(base_url=NER_SERVICE_URL, timeout=NER_TIMEOUT_S)
_active_url = NER_SERVICE_URL


def active_url() -> str:
    """Alamat NER Service yang sedang dipakai."""
    return _active_url


async def use_service(url: str) -> None:
    """Pindah ke NER Service lain saat berjalan. Client lama ditutup rapi supaya
    koneksinya tidak menggantung."""
    global _client, _active_url
    lama = _client
    _client = httpx.AsyncClient(base_url=url, timeout=NER_TIMEOUT_S)
    _active_url = url
    try:
        await lama.aclose()
    except RuntimeError:
        # Client lama dibuat di event loop lain (lazim saat pengujian). Menutupnya
        # hanya bersih-bersih; koneksinya akan dilepas saat objeknya dibuang.
        pass


_active_backend = os.getenv("NER_BACKEND", "spacy").lower()


def active_backend() -> str:
    """Backend NER yang sedang aktif ('spacy' atau 'indobert')."""
    return _active_backend


def set_backend(backend: str) -> None:
    """Ubah backend NER yang dipanggil."""
    global _active_backend
    _active_backend = backend.lower()


class NerUnavailable(Exception):
    """NER Service tidak bisa dihubungi / error. Callback yang memutuskan nasibnya."""


async def detect(text: str, backend: str | None = None) -> list[dict]:
    b = (backend or _active_backend).lower()
    try:
        resp = await _client.post("/ner", json={"text": text, "backend": b})
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
