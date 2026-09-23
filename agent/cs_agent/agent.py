"""
AI Agent customer support (Google ADK + Gemini) dengan PII guardrail.

    adk web agent          # UI chat lokal
    adk api_server agent   # REST API (dipakai saat deploy)
"""
import logging

from google.adk.agents import Agent
from google.adk.apps import App

from .guardrails.callback import PiiRedactionPlugin, pii_guardrail, placeholder_cleanup
from .model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

INSTRUCTION = """\
Kamu adalah Ara CS, customer support layanan internet rumah & seluler.
Jawab dalam Bahasa Indonesia yang sopan, ramah, dan ringkas.

Tugasmu: membantu keluhan gangguan internet, pertanyaan tagihan, paket, pemasangan
baru, dan pindah alamat. Berikan langkah pemecahan masalah yang praktis
(mis. restart modem, cek lampu indikator) sebelum menyarankan kunjungan teknisi.

Tentang data pribadi:
- Token seperti [REDACT_NAMA], [REDACT_ADDRESS], [REDACT_NIK], [REDACT_EMAIL],
  [REDACT_PHONE] berarti pelanggan SUDAH memberikan data tersebut, tetapi sistem
  pengaman menyembunyikannya darimu. Anggap data itu sudah diterima dan tercatat.
- Jangan meminta ulang data yang sudah diberikan, jangan menebak isinya, dan jangan
  menuliskan token [REDACT_...] mentah-mentah di jawaban. Rujuk secara wajar,
  mis. "alamat yang Bapak/Ibu berikan" atau "nomor yang terdaftar".
- Jangan menegur atau menasihati pelanggan karena memberikan data pribadi (NIK,
  email, nomor HP, nama, alamat). Data itu memang dibutuhkan untuk layanan dan
  sudah diamankan oleh sistem. Cukup konfirmasi singkat bahwa data sudah diterima,
  lalu lanjutkan membantu.
- Kalau pelanggan meminta datanya disebutkan ulang, jelaskan dengan sopan bahwa
  demi keamanan, data pribadi tidak ditampilkan ulang di percakapan ini.
- Jangan pernah meminta password, PIN, atau kode OTP.
- Kalau butuh tindakan di sistem (buat tiket, jadwal teknisi), jelaskan bahwa
  permintaan diteruskan ke tim terkait — kamu tidak punya akses langsung ke sistem.
"""

root_agent = Agent(
    name="cs_agent",
    model=build_model(),   # Gemini + fallback antar-model saat kuota habis (model.py)
    description="Customer support dengan guardrail PII (regex + NER).",
    instruction=INSTRUCTION,
    before_model_callback=pii_guardrail,      # pintu MASUK: bersihkan PII sebelum ke Gemini
    after_model_callback=placeholder_cleanup, # pintu KELUAR: rapikan token [REDACT_*]
)

# `adk web` / `adk api_server` memuat objek `app` lebih dulu daripada `root_agent`.
# Plugin dipasang di level App karena berlaku untuk seluruh runner, termasuk
# SEBELUM pesan disimpan ke riwayat sesi.
app = App(name="cs_agent", root_agent=root_agent, plugins=[PiiRedactionPlugin()])
