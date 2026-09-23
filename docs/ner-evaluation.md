# Evaluasi Model NER

> Model `pii_ner_id-3.3.0`. File ini ditulis ulang otomatis oleh `python ner_service/evaluate.py`.

Semua data uji ditulis tangan, dengan nama & alamat yang **tidak ada** di data latih. Sejarah iterasi dan catatan metodologi: [ner-iterasi.md](ner-iterasi.md).

## Skor

### test_v8 — `test_v8.jsonl` (40 kalimat, **buta** untuk model ini — batas alamat & frasa lokasi umum)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 6 | 6 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ADDRESS | 22 | 19 | 5 | 3 | 0.79 | 0.86 | 0.83 | 0.85 | 1.00 |
| TOTAL | 28 | 25 | 5 | 3 | 0.83 | 0.89 | 0.86 | 0.88 | 1.00 |

Kalimat tanpa PII yang tetap bersih: **16/18**

### test_v7 — `test_v7.jsonl` (40 kalimat, buta v9 — alamat tanpa awalan, angka utama saat model dipilih)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 6 | 5 | 2 | 1 | 0.71 | 0.83 | 0.77 | 0.81 | 1.00 |
| ADDRESS | 20 | 16 | 4 | 4 | 0.80 | 0.80 | 0.80 | 0.80 | 0.90 |
| TOTAL | 26 | 21 | 6 | 5 | 0.78 | 0.81 | 0.79 | 0.80 | 0.92 |

Kalimat tanpa PII yang tetap bersih: **18/20**

### test_v6 — `test_v6.jsonl` (40 kalimat, buta v8 — nama hari & bulan, pembanding)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 11 | 10 | 1 | 1 | 0.91 | 0.91 | 0.91 | 0.91 | 1.00 |
| ADDRESS | 9 | 9 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TOTAL | 20 | 19 | 1 | 1 | 0.95 | 0.95 | 0.95 | 0.95 | 1.00 |

Kalimat tanpa PII yang tetap bersih: **20/20**

### test_v5 — `test_v5.jsonl` (40 kalimat, buta v7 — posisi nama & alamat tanpa awalan, pembanding)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 14 | 13 | 1 | 1 | 0.93 | 0.93 | 0.93 | 0.93 | 1.00 |
| ADDRESS | 10 | 10 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TOTAL | 24 | 23 | 1 | 1 | 0.96 | 0.96 | 0.96 | 0.96 | 1.00 |

Kalimat tanpa PII yang tetap bersih: **20/20**

### test_v4 — `test_v4.jsonl` (30 kalimat, buta v6 — kalimat tanya & alamat tanpa awalan, pembanding)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 9 | 8 | 2 | 1 | 0.80 | 0.89 | 0.84 | 0.87 | 1.00 |
| ADDRESS | 6 | 6 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TOTAL | 15 | 14 | 2 | 1 | 0.88 | 0.93 | 0.90 | 0.92 | 1.00 |

Kalimat tanpa PII yang tetap bersih: **17/18**

### test_v3 — `test_v3.jsonl` (30 kalimat, buta v5 — kalimat curhat, pembanding)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 9 | 9 | 1 | 0 | 0.90 | 1.00 | 0.95 | 0.98 | 1.00 |
| ADDRESS | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TOTAL | 12 | 12 | 1 | 0 | 0.92 | 1.00 | 0.96 | 0.98 | 1.00 |

Kalimat tanpa PII yang tetap bersih: **19/20**

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
| PERSON | 12 | 9 | 3 | 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0.83 |
| ADDRESS | 10 | 9 | 1 | 1 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| TOTAL | 22 | 18 | 4 | 4 | 0.82 | 0.82 | 0.82 | 0.82 | 0.86 |

Kalimat tanpa PII yang tetap bersih: **4/5**

### val — `val.jsonl` (98 kalimat, dipakai memilih epoch — optimistis)

| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|---|---|---|---|
| PERSON | 29 | 28 | 1 | 1 | 0.97 | 0.97 | 0.97 | 0.97 | 1.00 |
| ADDRESS | 31 | 29 | 5 | 2 | 0.85 | 0.94 | 0.89 | 0.92 | 1.00 |
| TOTAL | 60 | 57 | 6 | 3 | 0.90 | 0.95 | 0.93 | 0.94 | 1.00 |

Kalimat tanpa PII yang tetap bersih: **43/46**

- **Exact match**: benar hanya bila label DAN batas awal-akhir persis sama.
- **F2**: F-score yang menimbang recall 2× — metrik pemilihan model, karena untuk guardrail FN (bocor) lebih mahal daripada FP (salah sensor).
- **Recall longgar**: entity asli yang setidaknya tersentuh tebakan berlabel sama.

## Kesalahan di test_v7

- **FN** = entity asli tidak tertebak dengan tepat → berpotensi bocor.
- **FP** = tebakan yang salah (bukan entity, atau batasnya meleset) → salah sensor.
- Satu entity yang batasnya meleset tercatat dua kali: FN (yang benar) + FP (yang ditebak).

| Jenis | Label | Potongan | Kalimat |
|---|---|---|---|
| FP | ADDRESS | `kota lain` | pindah rumah ke kota lain, layanan bisa ikut pindah? |
| FP | ADDRESS | `gang sempit` | rumah saya di gang sempit, teknisi bisa masuk tidak? |
| FP | ADDRESS | `jatinegara kaum jakarta` | tolong pasang di jatinegara kaum jakarta timur ya |
| FN | ADDRESS | `jatinegara kaum jakarta timur` | tolong pasang di jatinegara kaum jakarta timur ya |
| FP | PERSON | `tembalang bulusan semarang` | alamat saya sekarang tembalang bulusan semarang |
| FN | ADDRESS | `tembalang bulusan semarang` | alamat saya sekarang tembalang bulusan semarang |
| FP | PERSON | `bernama Tarmizi Alhabsyi` | pelanggan bernama Tarmizi Alhabsyi tinggal di pinang ranti jakarta timur |
| FN | PERSON | `Tarmizi Alhabsyi` | pelanggan bernama Tarmizi Alhabsyi tinggal di pinang ranti jakarta timur |
| FP | ADDRESS | `gunung sahari mangga dua jakarta` | kapan teknisi datang ke gunung sahari mangga dua jakarta pusat? |
| FN | ADDRESS | `gunung sahari mangga dua jakarta pusat` | kapan teknisi datang ke gunung sahari mangga dua jakarta pusat? |
| FN | ADDRESS | `kebon kacang jakarta` | tolong cek tagihan buat Rosmalinda Purbaningrum di kebon kacang jakarta |
