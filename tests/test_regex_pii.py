import pytest

from cs_agent.guardrails.regex_pii import redact


@pytest.mark.parametrize("text, expected", [
    # NIK
    ("NIK saya 3201234567890123 ya kak", "NIK saya [REDACT_NIK] ya kak"),
    ("NIK:3201234567890123ya", "NIK:[REDACT_NIK]ya"),               # tanpa spasi
    # Kasus uji user (22 Sep): huruf menempel di KIRI & KANAN — dengan \b ini lolos
    ("ohhya3276011310010012ini nik saya", "ohhya[REDACT_NIK]ini nik saya"),
    # Kasus uji user: salah ketik spasi ".comi ni" -> "comi" sah sebagai TLD, jadi
    # huruf "i" ikut tersensor. FP kecil yang disengaja: lebih baik kelebihan sensor.
    ("rugi@gmail.comi ni email saya", "[REDACT_EMAIL] ni email saya"),
    ("ID order 32012345678901234567", "ID order 32012345678901234567"),  # 20 digit: bukan NIK
    ("kode 123456789012345", "kode 123456789012345"),                 # 15 digit
    ("NIK 3201 2345 6789 0123 ya", "NIK [REDACT_NIK] ya"),           # 4-4-4-4 spasi
    ("NIK 3201-2345-6789-0123", "NIK [REDACT_NIK]"),                  # 4-4-4-4 strip
    ("NIK 320123 456789 0123", "NIK [REDACT_NIK]"),                   # 6-6-4
    ("HP 0812 3456 7890 ya", "HP [REDACT_PHONE] ya"),                 # 12 digit tetap telepon
    # Email
    ("email budi.s@gmail.com ya", "email [REDACT_EMAIL] ya"),
    ("kirim ke cs@mail.telkom.co.id", "kirim ke [REDACT_EMAIL]"),
    ("081234567890@gmail.com", "[REDACT_EMAIL]"),                    # email dulu, baru telepon
    # Telepon
    ("HP saya 081234567890", "HP saya [REDACT_PHONE]"),
    ("hubungi +6281234567890", "hubungi [REDACT_PHONE]"),
    ("wa 6281234567890 ya", "wa [REDACT_PHONE] ya"),
    ("nomor 0812-3456-7890", "nomor [REDACT_PHONE]"),
    ("nomor +62 812 3456 7890", "nomor [REDACT_PHONE]"),
    ("Tagihan saya Rp1508000", "Tagihan saya Rp1508000"),            # FP soal B sudah hilang
    # Tanpa PII
    ("internet saya mati sejak kemarin", "internet saya mati sejak kemarin"),
])
def test_redact(text, expected):
    assert redact(text)[0] == expected


def test_multiple_and_findings_have_no_values():
    text = "NIK 3201234567890123, email a@b.co, HP 081234567890"
    clean, findings = redact(text)
    assert clean == "NIK [REDACT_NIK], email [REDACT_EMAIL], HP [REDACT_PHONE]"
    assert [f.type for f in findings] == ["NIK", "EMAIL", "PHONE"]
    # Finding tidak boleh membawa nilai PII (aman di-log)
    assert not any(hasattr(f, "value") for f in findings)


# Telepon rumah (ditambahkan v9): kode area 2-3 digit, berbagai gaya penulisan.
@pytest.mark.parametrize("text,expected", [
    ("Telepon rumah saya 021-5551234", "Telepon rumah saya [REDACT_PHONE]"),
    ("hubungi (021) 5551234 ya kak", "hubungi [REDACT_PHONE] ya kak"),
    ("nomor rumah 0274 123456", "nomor rumah [REDACT_PHONE]"),
    ("bisa telepon +62 21 5551234", "bisa telepon [REDACT_PHONE]"),
    ("0361-9876543 itu nomor kantor", "[REDACT_PHONE] itu nomor kantor"),
])
def test_redact_telepon_rumah(text, expected):
    assert redact(text)[0] == expected


# Angka yang BUKAN telepon tidak boleh ikut tersensor.
@pytest.mark.parametrize("text", [
    "Tagihan saya Rp1508000 bulan ini, kok naik ya?",
    "Order ID 32012345678901234567 belum diproses",
    "paket 50 Mbps harga 350000 per bulan",
    "saya bayar 150000 lewat BCA tanggal 12",
])
def test_angka_bukan_telepon_tidak_disensor(text):
    assert redact(text)[0] == text
