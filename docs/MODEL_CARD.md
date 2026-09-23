# Model Card — `pii_ner_id` v3.3.0

## Ringkasan

| | |
|---|---|
| Tugas | Named Entity Recognition untuk guardrail PII |
| Label | `PERSON` (nama orang), `ADDRESS` (alamat) |
| Bahasa | Bahasa Indonesia, gaya chat customer support (formal & informal) |
| Arsitektur | spaCy 3.8 `ner` (transition-based parser + CNN tok2vec lebar 128, hash embedding 5000 baris **+ static word vectors**), tokenizer `id` |
| Bobot NER | Dilatih **dari nol** dengan data kita — tidak ada bobot NER pretrained |
| Fitur tambahan | fastText `cc.id.300` (Common Crawl, CC BY-SA 3.0): 200 ribu kata teratas dipangkas jadi 20 ribu vektor, **dibekukan** (hanya dibaca, tidak dilatih) |
| Pasca-proses | `ner_service/postprocess.py`: kata umum di tepi span PERSON dipangkas |
| Ukuran | ~40 MB di disk, ~150 MB RAM setelah dimuat |
| Latency (CPU, 1 thread) | p50 2,1 ms (≈60 karakter) – 12,7 ms (≈1.000 karakter), lihat [resource-performance.md](resource-performance.md) |
| Dipakai oleh | NER Service (`ner_service/app.py`) → dipanggil guardrail agent |

## Tujuan penggunaan

Menandai nama dan alamat di pesan pelanggan **sebelum** teks dikirim ke LLM, sebagai
lapis kedua setelah regex (NIK, email, telepon). Keluaran dipakai untuk **redaksi**,
bukan untuk ekstraksi data atau keputusan tentang pelanggan.

**Tidak cocok untuk:** dokumen panjang non-chat (kontrak, surat), bahasa selain Indonesia,
atau sebagai satu-satunya pengaman PII tanpa regex dan fail-closed.

## Data

| Set | Jumlah | Sumber | Peran |
|---|---|---|---|
| train | 3.600 kalimat | sintetis — 49 template × daftar nama/alamat + kalimat rakitan curhat/rupiah (`generate_train.py`, seed 42) | melatih bobot |
| dev | 400 kalimat | pecahan generator yang sama | memantau loss saja (skornya selalu ~1.00) |
| val | 32 kalimat | ditulis tangan | memilih epoch (F2) |
| **test_v7** | **40 kalimat** | **ditulis tangan, dikunci sebelum v9; 20 tanpa PII soal lokasi, 20 alamat tanpa awalan "Jl."** | **angka utama, dievaluasi sekali** |
| test_v6 | 40 kalimat | ditulis tangan, buta untuk v8; nama hari & bulan | pembanding |
| test_v5 | 40 kalimat | ditulis tangan, buta untuk v7; posisi nama & alamat tanpa awalan | pembanding |
| test_v4 | 30 kalimat | ditulis tangan, buta untuk v6; 18 kalimat tanya CS tanpa PII | pembanding |
| test_v3 | 30 kalimat | ditulis tangan, buta untuk v5; 20 kalimat curhat tanpa PII | pembanding |
| test_v2 | 20 kalimat | ditulis tangan, buta untuk v4 | pembanding |
| test_v1 | 20 kalimat | ditulis tangan | terkontaminasi (dilihat di iterasi v1–v3), hanya pembanding |

Data sintetis dipakai karena chat pelanggan asli tidak boleh digunakan (PII) dan
agar offset entity pasti benar. Nama/alamat di val & test tidak ada di daftar generator.
Keragaman nama: Jawa, Sunda, Batak (termasuk pola "br" marga), Bali, Minang,
Tionghoa-Indonesia, Arab-Indonesia, nama baptis. 30% kalimat latih di-lowercase.

## Pelatihan

20 epoch, dropout 0,3, minibatch 4→32, seed 1 (epoch 17). Konfigurasi dipilih dari **sweep**
14 konfigurasi × 3–8 seed (`ner_service/sweep.py`, VM GCP 16 vCPU) berdasarkan **rata-rata F2
di val** (86 kalimat, model + pasca-proses), lalu seed terbaik di val. Test tidak dipakai untuk
memilih apa pun — termasuk saat seed lain terbukti lebih baik di test.

## Hasil

| Set | Precision | Recall | F1 | F2 | Recall longgar |
|---|---|---|---|---|---|
| **test_v7 (buta)** | **0.78** | **0.81** | **0.79** | **0.80** | **0.92** |
| test_v6 | 0.95 | 0.95 | 0.95 | 0.95 | 1.00 |
| test_v5 | 0.96 | 0.96 | 0.96 | 0.96 | 1.00 |
| test_v4 | 0.88 | 0.93 | 0.90 | 0.92 | 1.00 |
| test_v3 | 0.92 | 1.00 | 0.96 | 0.98 | 1.00 |
| test_v2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| test_v1 (terkontaminasi) | 0.82 | 0.82 | 0.82 | 0.82 | 0.86 |
| val (86) | 0.96 | 0.98 | 0.97 | 0.98 | 1.00 |

**Recall longgar 1.00 di lima set** — tidak ada nama/alamat yang lolos utuh. Kalimat tanpa PII
yang tetap bersih: 18/20 test_v7, 20/20 test_v6 & test_v5, 17/18 test_v4, 19/20 test_v3.
Precision test_v7 turun karena batas alamat kadang meleset satu kata (dihitung FP + FN
sekaligus), bukan karena kebocoran.

Guardrail lengkap (regex + model ini) pada 22 kalimat campuran / 32 item PII:
**32 tertutup, 0 bocor**, 0 kata non-PII ikut tersensor — lihat
[guardrail-evaluation.md](guardrail-evaluation.md).

Rincian kesalahan: [ner-evaluation.md](ner-evaluation.md). Sejarah iterasi v1–v9 dan
temuan *test-set leakage*: [ner-iterasi.md](ner-iterasi.md).

## Batasan yang diketahui

- **Ukuran uji kecil.** 20 kalimat / ~20 entity per set: satu entity ≈ 5%. Angka di atas
  indikasi, bukan bukti statistik.
- **Bias penulis.** Train, val, dan test ditulis pihak yang sama; gaya kalimat bisa mirip.
  Uji paling jujur adalah chat asli yang sudah dianonimkan.
- **Batas alamat yang langsung disambung keluhan** kadang kelebihan
  (`… Semarang lampu LOS merah`) — aman dari sisi kebocoran, tapi menyensor kata ekstra.
- **Kata tunggal huruf kecil** kadang ditebak PERSON (`pending`, `merah`, `jelek`) — FP.
- **Bias posisi dari template**: kata apa pun tepat setelah "Saya" cenderung ditebak PERSON
  (`Saya memberikan…` → "memberikan", `Saya ingin belajar budi pekerti` → "belajar budi pekerti"),
  karena di data latih "Saya {nama}" sangat dominan. FP, bukan kebocoran. Ditemukan saat
  mencoba model secara manual. **v5 menguranginya** dengan kalimat curhat rakitan ("aku lagi …",
  "saya sudah …"): kalimat tanpa PII yang bersih di test_v3 naik 8/20 → 14/20. Sisa FP ada pada
  kata yang belum pernah dilihat model (`males ribet`, `ngerasa`) — batas model tanpa embedding.
- **Simbol mata uang**: v4 menebak `Rp` (dari `Rp1508000`) sebagai PERSON. v5 dilatih dengan
  kalimat bernominal; aturan di `ner_client.plausible_entities` tetap dipertahankan sebagai cadangan.
- **Nama yang lolos utuh di v5**: `komang adi putra` (v4 menangkap sebagian), `mulyadi`,
  `yohanes ndruru`; `Jl. Ir. H. Juanda …` hanya tertutup separuh. v5 menukar sedikit recall
  dengan salah sensor yang jauh lebih sedikit — lihat [ner-iterasi.md](ner-iterasi.md).
- Tidak mengenali entity lain: nomor rekening, tanggal lahir, nama perusahaan, plat nomor.

## Mitigasi di sistem

- Regex menangani PII berpola pasti **sebelum** model.
- Aturan kewarasan: tebakan `PERSON` tanpa huruf dibuang; tebakan yang menimpa placeholder regex diabaikan.
- Fail-closed: bila NER Service tidak tersedia, pesan ditahan dan LLM tidak dipanggil.
- CI menjalankan evaluasi end-to-end dan **gagal** bila ada PII yang bocor utuh.

## Etika & privasi

- Tidak ada data pelanggan asli dalam pelatihan maupun pengujian.
- NER Service tidak me-log isi teks; metrik Prometheus hanya angka agregat.
- Kesalahan model berdampak asimetris: FN = data pribadi terkirim ke pihak ketiga (LLM),
  FP = jawaban CS sedikit kurang personal. Karena itu pemilihan model memakai F2.
