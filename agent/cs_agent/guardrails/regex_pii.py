"""
Guardrail PII lapis 1: regex untuk PII yang bentuknya pasti (NIK, email, telepon).

Nama & alamat TIDAK ditangani di sini — bentuknya tidak pasti, jadi diserahkan ke
model NER (lihat ner_client.py). Regex = aturan tulisan tangan, bukan model.
"""
import re
from dataclasses import dataclass

# Urutan PENTING — dijalankan dari atas ke bawah pada teks yang sudah diredaksi
# oleh pola sebelumnya:
#   1. NIK dulu: 16 digit utuh harus habis duluan, supaya potongannya tidak
#      disangka nomor telepon (kode provinsi Kalteng berawalan 62).
#   2. Email sebelum telepon: "081234567890@gmail.com" harus tertangkap utuh
#      sebagai email, bukan dipotong jadi telepon + sisa "@gmail.com".
PATTERNS: list[tuple[str, re.Pattern]] = [
    # (?<!\d) dan (?!\d) = "pagar": kiri-kanan tidak boleh angka.
    # Baseline PDF memakai ^...$ yang hanya cocok bila SELURUH pesan adalah NIK.
    # \b tidak dipakai karena huruf dihitung bagian kata: "0123ya" akan lolos.
    # Juga menangkap penulisan berkelompok yang lazim di chat:
    #   3201 2345 6789 0123 / 3201-2345-6789-0123 (4-4-4-4)
    #   320123 456789 0123                       (6-6-4 = wilayah, tgl lahir, urut)
    # Struktur NIK (kode provinsi, tanggal lahir) sengaja TIDAK divalidasi: salah ketik
    # satu digit akan lolos (FN = bocor). Untuk guardrail, recall > precision.
    ("NIK", re.compile(r"(?<!\d)(?:\d{4}[ .-]?\d{4}[ .-]?\d{4}[ .-]?\d{4}"
                       r"|\d{6}[ .-]\d{6}[ .-]\d{4})(?!\d)")),

    # Baseline PDF: titik sebelum TLD tidak di-escape (".") sehingga cocok dengan
    # karakter apa pun. Di sini "\." = titik sungguhan.
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),

    # Baseline PDF: "(+62|62|0)" -> "+" tidak di-escape = error "nothing to repeat".
    # Tambahan: boleh ada spasi/strip di antara angka (0812-3456-7890, +62 812 3456 7890).
    ("PHONE", re.compile(r"(?<!\d)(?:\+62|62|0)[\s-]?8[1-9](?:[\s-]?\d){6,10}(?!\d)")),

    # Telepon rumah — di luar contoh soal, tapi pelanggan internet rumah sering
    # memberikannya. Kode area 2-3 digit (021, 0274, ...) lalu 6-8 digit, boleh
    # dipisah spasi/strip, boleh dalam kurung: (021) 5551234.
    # Dijalankan SETELAH pola HP, jadi nomor HP tidak tersentuh pola ini.
    # Sengaja longgar: untuk guardrail, nomor rekening yang ikut tersensor jauh
    # lebih murah daripada nomor telepon yang lolos.
    ("PHONE", re.compile(r"(?<!\d)(?:\(0\d{2,3}\)|(?:\+62|62|0)[\s-]?\d{2,3})"
                         r"[\s-]?\d{3}[\s-]?\d{3,5}(?!\d)")),
]

PLACEHOLDER = {
    "NIK": "[REDACT_NIK]",
    "EMAIL": "[REDACT_EMAIL]",
    "PHONE": "[REDACT_PHONE]",
}


@dataclass
class Finding:
    """Satu temuan PII. Sengaja TIDAK menyimpan nilai aslinya — aman untuk di-log.
    start/end = posisi karakter di teks ASLI (end eksklusif)."""
    type: str
    start: int
    end: int


# Karakter penutup: bukan angka/huruf/@, jadi pola berikutnya "melihat" bagian yang
# sudah tertangkap sama seperti melihat placeholder — tapi panjang teks tidak berubah,
# sehingga posisi tetap mengacu ke teks asli.
_MASK = "\x00"


def find(text: str) -> list[Finding]:
    """Temukan NIK/email/telepon, urut posisi, dalam koordinat teks asli."""
    findings: list[Finding] = []
    masked = text
    for pii_type, pattern in PATTERNS:
        for m in pattern.finditer(masked):
            findings.append(Finding(pii_type, m.start(), m.end()))
        for f in findings:
            masked = masked[:f.start] + _MASK * (f.end - f.start) + masked[f.end:]
    return sorted(findings, key=lambda f: f.start)


def replace_spans(text: str, spans: list[tuple[int, int, str]], placeholder: dict) -> str:
    """Ganti span (start, end, type) dengan placeholder — dari belakang, supaya offset
    span sebelumnya tidak bergeser."""
    for start, end, t in sorted(spans, key=lambda s: s[0], reverse=True):
        text = text[:start] + placeholder[t] + text[end:]
    return text


def redact(text: str) -> tuple[str, list[Finding]]:
    """Ganti NIK/email/telepon dengan placeholder."""
    findings = find(text)
    return replace_spans(text, [(f.start, f.end, f.type) for f in findings], PLACEHOLDER), findings
