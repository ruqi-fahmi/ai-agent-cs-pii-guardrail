"""
Kunci perilaku kalimat-kalimat yang dipakai saat demo.

Kenapa ada: tiap iterasi model memperbaiki satu hal dan sesekali merusak hal lain
(v7 menghilangkan kebocoran alamat tapi mulai menyensor nama hari; v8 sebaliknya).
Test ini menjadikan kasus-kasus itu **regresi yang gagal keras**, bukan temuan
yang baru ketahuan saat demo di depan penilai.

Dijalankan tanpa API key: model NER dimuat in-process, Gemini tidak dipanggil.
"""
import sys
from pathlib import Path

import pytest
import spacy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ner_service"))

from cs_agent.guardrails import ner_client                      # noqa: E402
from cs_agent.guardrails.regex_pii import redact as regex_redact  # noqa: E402
from postprocess import clean_ents                              # noqa: E402


@pytest.fixture(scope="module")
def redact():
    """Pipeline yang sama dengan service: regex -> NER -> postprocess -> ganti span."""
    nlp = spacy.load(ROOT / "ner_service" / "model")

    def run(text: str) -> str:
        clean, _ = regex_redact(text)
        ents = [{"label": e.label_, "text": e.text, "start": e.start_char, "end": e.end_char}
                for e in clean_ents(nlp(clean))]
        return ner_client.apply_entities(clean, ents)[0]
    return run


# (pesan, potongan yang HARUS hilang, placeholder yang harus muncul)
HARUS_TERSENSOR = [
    ("Halo kak, saya Budi Santoso, NIK 3201234567890123. Internet di rumah saya "
     "Jl. Merdeka No 5 Bandung mati sejak kemarin",
     ["Budi Santoso", "3201234567890123", "Merdeka"],
     ["[REDACT_NAMA]", "[REDACT_NIK]", "[REDACT_ADDRESS]"]),
    ("Kalau teknisi mau datang hubungi 0812-3456-7890 atau email budi.s@gmail.com ya",
     ["0812-3456-7890", "budi.s@gmail.com"],
     ["[REDACT_PHONE]", "[REDACT_EMAIL]"]),
    # v9: alamat tanpa kata "Jalan" — kebocoran utama versi sebelumnya
    ("tolong pasang di jatinegara kaum jakarta timur ya, atas nama fahmi",
     ["jatinegara", "fahmi"],
     ["[REDACT_ADDRESS]", "[REDACT_NAMA]"]),
]

# Kalimat tanpa PII: harus lewat UTUH. Dua di antaranya adalah kegagalan nyata
# versi lama (v7 menyensor "Sabtu"/"Agustus", v5 menyensor frasa curhat).
HARUS_UTUH = [
    "Apakah teknisi bisa datang hari Sabtu? tagihan Agustus saya belum lunas",
    "apakah status langganan saya masih aktif?",
    "aku lagi pusing sama pola pikir orang",
    "Tagihan saya Rp1508000 bulan ini, kok naik ya?",
]


@pytest.mark.parametrize("text,hilang,muncul", HARUS_TERSENSOR,
                         ids=["nik-nama-alamat", "telepon-email", "alamat-tanpa-awalan"])
def test_pii_demo_tersensor(redact, text, hilang, muncul):
    out = redact(text)
    for frag in hilang:
        assert frag not in out, f"{frag!r} masih ada di {out!r}"
    for ph in muncul:
        assert ph in out, f"{ph} tidak muncul di {out!r}"


@pytest.mark.parametrize("text", HARUS_UTUH)
def test_kalimat_tanpa_pii_tidak_disensor(redact, text):
    assert redact(text) == text


# Kata peran setelah pemicu "atas nama" sempat ditandai PERSON (ditemukan saat mencoba
# model secara manual). Yang PII di kalimat itu namanya, bukan kata perannya.
@pytest.mark.parametrize("peran", ["perusahaan", "suami", "istri", "yayasan", "koperasi",
                                   "almarhum", "saudara", "pengurus"])
def test_kata_peran_tidak_disensor(redact, peran):
    teks = f"atas nama {peran}, mau komplain soal tagihan"
    assert redact(teks) == teks


def test_nama_setelah_kata_peran_tetap_disensor(redact):
    out = redact("atas nama suami saya, Budi Santoso, mau komplain")
    assert "Budi Santoso" not in out
    assert "[REDACT_NAMA]" in out


def test_nama_bali_anak_agung_tidak_ikut_terpangkas(redact):
    # "anak" sengaja tidak masuk daftar kata umum: Anak Agung adalah nama sungguhan.
    assert "Anak Agung" not in redact("atas nama Anak Agung Rai, mau pasang baru")


# Singkatan gaya chat sempat ditandai PERSON karena daftar stopword spaCy hanya memuat
# bentuk bakunya. Ditemukan saat demo: "apa yg bisa dibantu?" -> "yg" tersensor.
@pytest.mark.parametrize("text", [
    "gangguan internet apa yg bisa dibantu?",
    "sy mau tanya tagihan bln ini",
    "tlg dicek dgn segera ya",
    "kmu bisa bantu cek status pemasangan?",
])
def test_singkatan_chat_tidak_disensor(redact, text):
    assert redact(text) == text


def test_nama_setelah_singkatan_tetap_disensor(redact):
    out = redact("sy budi santoso mau komplain")
    assert "budi santoso" not in out and "[REDACT_NAMA]" in out
