# Evaluasi Model NER

> Model `pii_ner_id-3.2.0`. File ini ditulis ulang otomatis oleh `python ner_service/evaluate.py`.

Semua data uji ditulis tangan, dengan nama & alamat yang **tidak ada** di data latih. Sejarah iterasi dan catatan metodologi: [ner-iterasi.md](ner-iterasi.md).

## Skor

### test_v6 — `test_v6.jsonl` (40 kalimat, **buta** v8 — nama hari & bulan, angka utama)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 11 | 10 | 1 | 1 | 0.91 | 0.91 | 0.91 | 0.91 | 1.00 |
| ADDRESS | 9 | 6 | 1 | 3 | 0.86 | 0.67 | 0.75 | 0.70 | 0.78 |
| TOTAL | 20 | 16 | 2 | 4 | 0.89 | 0.80 | 0.84 | 0.82 | 0.90 |

Kalimat tanpa PII yang tetap bersih: **20/20**

### test_v5 — `test_v5.jsonl` (40 kalimat, buta v7 — posisi nama & alamat tanpa awalan, pembanding)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 14 | 14 | 1 | 0 | 0.93 | 1.00 | 0.97 | 0.99 | 1.00 |
| ADDRESS | 10 | 9 | 0 | 1 | 1.00 | 0.90 | 0.95 | 0.92 | 0.90 |
| TOTAL | 24 | 23 | 1 | 1 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 |

Kalimat tanpa PII yang tetap bersih: **20/20**

### test_v4 — `test_v4.jsonl` (30 kalimat, buta v6 — kalimat tanya & alamat tanpa awalan, pembanding)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 9 | 8 | 0 | 1 | 1.00 | 0.89 | 0.94 | 0.91 | 0.89 |
| ADDRESS | 6 | 4 | 0 | 2 | 1.00 | 0.67 | 0.80 | 0.71 | 0.67 |
| TOTAL | 15 | 12 | 0 | 3 | 1.00 | 0.80 | 0.89 | 0.83 | 0.80 |

Kalimat tanpa PII yang tetap bersih: **18/18**

### test_v3 — `test_v3.jsonl` (30 kalimat, buta v5 — kalimat curhat, pembanding)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 9 | 9 | 2 | 0 | 0.82 | 1.00 | 0.90 | 0.96 | 1.00 |
| ADDRESS | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TOTAL | 12 | 12 | 2 | 0 | 0.86 | 1.00 | 0.92 | 0.97 | 1.00 |

Kalimat tanpa PII yang tetap bersih: **18/20**

### test_v2 — `test_v2.jsonl` (20 kalimat, buta v4 — sudah dilihat sekali, pembanding)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 10 | 10 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ADDRESS | 9 | 9 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TOTAL | 19 | 19 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Kalimat tanpa PII yang tetap bersih: **5/5**

### test_v1 — `test.jsonl` (20 kalimat, terkontaminasi — pembanding)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 12 | 12 | 1 | 0 | 0.92 | 1.00 | 0.96 | 0.98 | 1.00 |
| ADDRESS | 10 | 9 | 0 | 1 | 1.00 | 0.90 | 0.95 | 0.92 | 0.90 |
| TOTAL | 22 | 21 | 1 | 1 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |

Kalimat tanpa PII yang tetap bersih: **4/5**

### val — `val.jsonl` (74 kalimat, dipakai memilih epoch — optimistis)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 27 | 26 | 0 | 1 | 1.00 | 0.96 | 0.98 | 0.97 | 0.96 |
| ADDRESS | 17 | 17 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TOTAL | 44 | 43 | 0 | 1 | 1.00 | 0.98 | 0.99 | 0.98 | 0.98 |

Kalimat tanpa PII yang tetap bersih: **36/36**

- **Exact match**: benar hanya bila label DAN batas awal-akhir persis sama.
- **F2**: F-score yang menimbang recall 2× — metrik pemilihan model, karena untuk guardrail FN (bocor) lebih mahal daripada FP (salah sensor).
- **Recall longgar**: entity asli yang setidaknya tersentuh tebakan berlabel sama.

## Kesalahan di test_v6

- **FN** = entity asli tidak tertebak dengan tepat → berpotensi bocor.
- **FP** = tebakan yang salah (bukan entity, atau batasnya meleset) → salah sensor.
- Satu entity yang batasnya meleset tercatat dua kali: FN (yang benar) + FP (yang ditebak).

| Jenis | Label | Potongan | Kalimat |
|---|---|---|---|
| FN | ADDRESS | `ujung menteng jakarta timur` | sejak Maret saya tinggal di ujung menteng jakarta timur |
| FN | ADDRESS | `cibaduyut bandung selatan` | aku pindah ke cibaduyut bandung selatan bulan April, gimana prosesnya? |
| FP | PERSON | `kristina` | tolong info tagihan Juli untuk ibu kristina boro |
| FN | PERSON | `kristina boro` | tolong info tagihan Juli untuk ibu kristina boro |
| FP | ADDRESS | `Griya Asri Permai Blok B3 Sidoarjo bulan Mei` | Pemasangan baru di Griya Asri Permai Blok B3 Sidoarjo bulan Mei |
| FN | ADDRESS | `Griya Asri Permai Blok B3 Sidoarjo` | Pemasangan baru di Griya Asri Permai Blok B3 Sidoarjo bulan Mei |
