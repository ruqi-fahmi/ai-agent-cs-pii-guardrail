# Evaluasi Guardrail End-to-End

> Ditulis ulang otomatis oleh `python scripts/eval_guardrail.py`.

Pipeline lengkap (regex → NER → redaksi) dijalankan pada [`scripts/guardrail_eval.jsonl`](../scripts/guardrail_eval.jsonl): 20 kalimat chat CS campuran, 29 item PII, termasuk kalimat tanpa PII yang mirip PII (nominal tagihan, order ID 20 digit, nama kota). Nama & alamat di set ini tidak ada di data latih maupun set uji NER. Mode: model NER in-process.

Pertanyaan yang dijawab: **berapa PII yang benar-benar sampai ke LLM?** — bukan sekadar akurasi model.

## Hasil

| Jenis | Lapis | Jumlah | Tertutup | Sebagian | Bocor utuh |
|---|---|---|---|---|---|
| NIK | regex | 4 | 4 | 0 | 0 |
| EMAIL | regex | 3 | 3 | 0 | 0 |
| PHONE | regex | 4 | 4 | 0 | 0 |
| PERSON | NER | 10 | 10 | 0 | 0 |
| ADDRESS | NER | 8 | 8 | 0 | 0 |
| **Total** | | **29** | **29** (100%) | **0** | **0** (0%) |

- **Tertutup**: tidak ada bagian nilai asli yang tersisa.
- **Sebagian**: sebagian token tersisa (mis. marga tertinggal) — bocor parsial.
- **Bocor utuh**: nilai asli masih lengkap di teks yang dikirim ke LLM.

## Item yang tidak tertutup penuh

_Tidak ada._

## Over-redaction — kata non-PII yang ikut tersensor (1)

| Kata | Kalimat asli |
|---|---|
| `order` | Order ID saya 32012345678901234567, statusnya masih pending |
