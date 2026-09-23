# Riwayat Iterasi Model NER

## v9 — alamat tanpa awalan "Jl." (model yang dipakai: `pii_ner_id-3.3.0`)

**Pemicu:** kekurangan utama v8 — alamat berbentuk kelurahan/kecamatan + kota tanpa kata
"Jalan" (`cibaduyut bandung selatan`, `ujung menteng jakarta timur`) lolos utuh ke LLM.

1. **`test_v7.jsonl` dikunci** (40 kalimat: 20 tanpa PII soal lokasi/pindah alamat — kategori
   yang paling rawan salah sensor kalau model dibuat lebih agresif soal alamat; 20 dengan
   alamat tanpa awalan, sebagian bercampur nama).
2. **val 74 → 86 kalimat** (+8 alamat tanpa awalan, +4 negatif soal lokasi).
3. **Data v9** (`--v8` membuat ulang data v8 byte-per-byte): porsi alamat tanpa awalan
   15% → 25%, 53 daerah tambahan, akhiran arah Jawa (Kidul/Lor/Wetan/Kulon), dan bentuk
   tiga bagian (kelurahan + kecamatan + kota).
4. **Sweep** 2 varian vektor × 8 seed, VM `n2d-highcpu-16` (~20 menit, ± Rp 4 ribu, dihapus).

| Konfigurasi | seed | val F2 rata-rata | median semua epoch | val bersih |
|---|---|---|---|---|
| **M: data v9 + vektor cased 20k + tok2vec lebar** | 8 | **0.964** | **0.910** | **39.4/40** |
| N: data v9 + vektor huruf kecil 20k + lebar | 8 | 0.922 | 0.849 | 38.5/40 |

Kunci kapital menang lagi, seperti v8. Terpilih **seed 1** dari val (F2 0.977).

test_v7 dibuka **sekali**. Yang dibandingkan bukan hanya F2, tapi **berapa entity yang lolos
utuh** di seluruh test set — metrik yang paling penting untuk guardrail:

| Model | entity lolos utuh (6 set) | test_v7 longgar / bersih | test_v6 F2 | test_v5 F2 | e2e over-redaction |
|---|---|---|---|---|---|
| v8 | **±10** | 0.85 / 18-20 | 0.82 | 0.96 | 1 kata |
| **v9 (seed 1)** | **±2** | **0.92** / 18-20 | **0.95** | 0.96 | **0 kata** |

Recall longgar **1.00 di test_v2–test_v6** (tidak ada nama/alamat yang lolos utuh di lima set
sekaligus), dan guardrail end-to-end: **29/29 PII tertutup, 0 bocor, 0 kata non-PII ikut
tersensor** — versi terbersih sejauh ini.

**Sisa kesalahan (jujur):** di test_v7 sebagian batas alamat meleset satu kata
(`jatinegara kaum jakarta` dari gold `jatinegara kaum jakarta timur`, `gunung sahari mangga
dua jakarta` dari `… jakarta pusat`). Alamatnya **tetap tersensor**, hanya potongannya kurang
panjang — tercatat sebagai FP+FN sekaligus, sehingga precision test_v7 turun ke 0.78 meski
kebocorannya justru berkurang. Dua frasa lokasi umum (`kota lain`, `gang sempit`) ditandai
ADDRESS; ini efek samping membuat model lebih peka terhadap alamat tanpa awalan.

**Catatan kejujuran:** sebagian kata daerah di test_v7 (`antapani`, `tembalang`, `panakkukang`)
juga ada di daftar generator, meski gabungan frasanya baru. Jadi test_v7 mengukur generalisasi
ke **kombinasi** yang belum dilihat, bukan ke kata yang sepenuhnya asing.

---

## v8 — nama hari & bulan (`pii_ner_id-3.2.0`)

**Pemicu:** v7 menandai `Sabtu`, `Rabu`, `Agustus`, `info`, `selesai` sebagai PERSON. Data latih
tidak pernah memuat nama hari/bulan, sehingga bagi model itu hanya "kata berkapital di tengah
kalimat" — bentuk yang selama ini berarti nama. val juga **tidak punya satu pun** kalimat
seperti itu, jadi val tidak bisa membedakan seed yang rawan.

1. **`test_v6.jsonl` dikunci** (40 kalimat: 20 tanpa PII berisi hari/bulan/kata CS, 20 dengan
   nama & alamat **plus** keterangan waktu — supaya model tidak belajar "ada nama hari berarti
   tidak ada nama orang").
2. **val 62 → 74 kalimat**: 8 negatif hari/bulan + 4 positif bercampur waktu.
3. **Data v8**: 18 template hari/bulan (`generate_train.py`, `--v7` membuat ulang data v7
   byte-per-byte) → 327 kalimat latih memuat hari/bulan.
4. **Sweep** 2 konfigurasi terbaik v7 × **8 seed**, VM `n2d-highcpu-16` (~20 menit, ± Rp 4 ribu,
   VM dihapus).

| Konfigurasi | seed | val F2 rata-rata | median semua epoch | val bersih |
|---|---|---|---|---|
| **L: data v8 + vektor cased 20k + tok2vec lebar** | 8 | **0.962** | **0.919** | **36/36 (semua seed)** |
| K: data v8 + vektor huruf kecil 20k + lebar | 8 | 0.933 | 0.878 | 35/36 |

Menarik: di sweep v7 kunci huruf kecil ≈ kapital; dengan data v8 justru **kapital yang menang**.
Dugaan (belum diuji): kapitalisasi ikut jadi petunjuk membedakan "Agustus" (bulan) dari nama.
Terpilih **seed 4** dari val (F2 0.982).

Lalu test_v6 dibuka **sekali**:

| Model | test_v6 F2 / longgar / bersih | test_v5 | test_v4 | test_v3 | e2e (29 PII) |
|---|---|---|---|---|---|
| v7 | 0.75 / 0.95 / **15/20** | 0.92 / 1.00 / 16/20 | 0.93 / 1.00 / 18/18 | 0.95 / 1.00 / 17/20 | 29 tertutup, 2 kata ikut tersensor |
| **v8 (seed 4)** | **0.82** / 0.90 / **20/20** | **0.96** / 0.96 / **20/20** | 0.83 / 0.80 / 18/18 | **0.97** / 1.00 / 18/20 | 29 tertutup, **1** kata ikut tersensor |
| v8, 8 seed | F2 0.80–0.95, **bersih 20/20 di semua seed** | | | | |

**Masalah hari/bulan hilang sepenuhnya**: seluruh 8 seed mendapat 20/20 kalimat bersih di
test_v6, dan test_v5 naik 16/20 → 20/20.

**Harga yang dibayar (jujur):** kebocoran naik. v7 hampir tidak punya FN utuh; v8 seed 4
melewatkan ±5 entity di seluruh test set — hampir semuanya **alamat tanpa awalan**
(`ujung menteng jakarta timur`, `cibaduyut bandung selatan`, `tamalanrea makassar`).
Untuk guardrail ini arah yang salah, dan **val lagi-lagi tidak memperingatkan**: seed 4 justru
seed dengan val F2 tertinggi, sementara 7 seed lain punya recall longgar 1.00 di test_v6.
Seed lain **tidak dipakai** — memilih seed berdasarkan test membuat angkanya palsu.

**Pelajaran yang terbawa ke v9:** val harus memuat kategori yang ingin dijaga, dalam jumlah
cukup. Sekarang val hanya punya segelintir alamat tanpa awalan, sehingga tidak bisa memilih
seed berdasarkan risiko kebocoran kategori itu. Langkah berikutnya: perbanyak alamat tanpa
awalan di val **dan** di data latih, kunci test_v7, sweep ulang.

---

## v7 — sweep konfigurasi × seed di VM GCP (`pii_ner_id-3.1.0`)

**Pemicu:** pemilik project merasa v6 belum cukup; sisa kebocoran v6 (`ciputat tangerang
selatan`, `agustinus lamere`) dan batas alamat yang kelebihan (`fatmawati jakarta, apakah`).

Urutan:

1. **`test_v5.jsonl` dikunci dulu** (40 kalimat: 20 tanpa PII, 20 dengan nama sesudah
   "pelanggan/buat/punya/an.", alamat tanpa awalan + arah mata angin, alamat disambung
   pertanyaan). Nama dicek otomatis tidak ada di generator. test_v4 sudah dibuka di v6,
   jadi tidak lagi dipakai untuk keputusan.
2. **val diperbesar 32 → 62 kalimat** (28 tanpa PII). Val 32 kalimat membuat skor
   loncat 0,73–0,96 antar-epoch: memilih model jadi lempar dadu.
3. **Data v7** (`generate_train.py`, `--v6` membuat ulang data lama byte-per-byte): 24
   template baru (posisi nama & alamat disambung pertanyaan, kata-kata sengaja berbeda
   dari val/test), 25 nama daerah tambahan, arah mata angin pada kota — kota/wilayah
   *saja* tetap bukan alamat.
4. **Vektor berkunci huruf kecil** (`prepare_vectors.py --attr LOWER`): fastText menyimpan
   "Fatmawati", "Agustinus", "Margonda" hanya dalam bentuk kapital, sehingga di chat huruf
   kecil kata itu tanpa vektor.
5. **Sweep** (`ner_service/sweep.py`) di VM `n2d-highcpu-16` (asia-southeast2, ~35 menit,
   ± Rp 7 ribu, VM dihapus setelahnya). Tiap konfigurasi 3 seed × 20 epoch, dipilih dari
   **rata-rata seed di val** (model + pasca-proses), bukan run terbaik.

| Konfigurasi | seed | val F2 rata-rata | median semua epoch | val bersih |
|---|---|---|---|---|
| **E: data v7 + vektor lower 20k + tok2vec lebar (128/5000)** | 6 | **0.962** | **0.914** | 27.8/28 |
| J: data v7 + vektor cased 20k + lebar | 3 | 0.960 | 0.914 | 28/28 |
| C: data v7 + vektor cased 20k | 3 | 0.951 | 0.915 | 27.3/28 |
| B: data v6 + vektor lower 20k | 6 | 0.949 | 0.872 | 28/28 |
| G: data v7 + vektor lower 50k + lebar | 3 | 0.950 | 0.883 | 27.3/28 |
| D: data v7 + vektor lower 50k | 3 | 0.945 | 0.868 | 27.7/28 |
| I: data v6 + vektor lower 20k + lebar | 3 | 0.941 | 0.884 | 26.7/28 |
| F: data v7 + vektor lower 20k + dropout 0.2 | 3 | 0.938 | 0.868 | 27.7/28 |
| A: data v7 + vektor lower 20k | 3 | 0.921 | 0.872 | 28/28 |
| H: data v7 tanpa vektor | 3 | 0.858 | 0.797 | 23.3/28 |
| *v6 yang ter-deploy* | 1 | *0.905* | | *27/28* |

Yang terbukti: vektor tetap faktor terbesar (H paling buruk); data v7 membantu **hanya bersama
tok2vec lebar** (I 0.941 vs E 0.962); 50 ribu vektor tidak lebih baik dari 20 ribu; kunci
huruf kecil vs kapital **seri** (E ≈ J) — lower dipilih karena alasan desain, bukan angka.
Dari 6 seed E, **seed 2** dipilih di val (F2 0.990, recall 1.00).

Lalu test_v5 dibuka **sekali** — untuk model terpilih, keenam seed E (sebaran), dan v6:

| Model | test_v5 P / R / F2 | longgar | bersih | test_v4 F2 / bersih | test_v3 F2 / bersih | bocor utuh (semua test) |
|---|---|---|---|---|---|---|
| v6 | 0.82 / 0.75 / 0.76 | 0.75 | 19/20 | 0.82 / 18/18 | 1.00 / 20/20 | **7** |
| **v7 (E seed 2)** | **0.79 / 0.96 / 0.92** | **1.00** | 16/20 | 0.93 / 18/18 | 0.95 / 17/20 | **0** |
| E, 6 seed | F2 0.92–1.00 (rata-rata 0.95) | 0.92–1.00 | 16–20/20 | | | |

**Harga yang dibayar (jujur):**
- Salah sensor naik lagi: seed 2 menandai nama hari/bulan (`Sabtu`, `Rabu`, `Agustus`), `info`,
  `selesai` sebagai PERSON. Guardrail e2e: 29/29 tertutup, tapi 2 kata non-PII ikut tersensor
  (v6: 0). Seed lain ada yang 20/20 bersih di test_v5 — **tidak dipakai**, karena memilih
  seed berdasarkan test = test-set leakage; angka 1.00 itu akan palsu.
- Pelajaran: sebaran antar-seed di test lebar (bersih 16–20/20) dan val **tidak punya satu
  pun kalimat berisi nama hari/bulan**, sehingga val tidak bisa membedakan seed yang rawan.
  Iterasi berikut: tambah kategori itu ke val & generator, kunci test_v6 dulu.
- Sweep dijalankan di Linux; hasil latih ulang di Windows tidak dijamin identik bit-per-bit.
  Model yang dipakai = artefak dari VM, konfigurasinya tercatat di `model/training_log.json`.

---

## v6 — word vectors fastText + pangkas kata umum (`pii_ner_id-3.0.0`)

**Pemicu (22 Sep, demo di GKE):** `...atas nama fahmi dengan alamat di fatmawati jakarta,
apakah status masih aktif?` → `apakah status` disensor sebagai nama, sedangkan
`fatmawati jakarta` **lolos ke Gemini**. Dua masalah dengan akar berbeda:

- **Salah sensor** — batas yang sudah tercatat di v5: model tanpa embedding tidak tahu arti
  kata, sehingga "apakah" dan "fahmi" sama-sama deretan huruf asing baginya.
- **Kebocoran alamat** — semua alamat di data latih v5 diawali Jalan/Gang/Perumahan/Kampung/...,
  jadi model tidak pernah melihat alamat tanpa awalan.

Urutan, mengikuti disiplin v4/v5:

1. Tulis **`test_v4.jsonl`** (30 kalimat: 18 kalimat tanya/permintaan CS tanpa PII, 12 kalimat
   dengan nama baru & alamat tanpa awalan) **sebelum** apa pun diubah. Baseline v5:
   **P 0.42 / R 0.53**, 12/18 kalimat bersih, 4 dari 6 alamat tanpa awalan bocor utuh.
2. **Pasca-proses** (`ner_service/postprocess.py`): kata umum di tepi span PERSON dipangkas
   (stopword Bahasa Indonesia bawaan spaCy + ~40 kosakata chat/CS). Dipakai di service,
   `evaluate.py`, dan `eval_guardrail.py`, jadi angka = yang benar-benar dikirim ke agent.
   Tes memastikan tidak ada satu pun nama di generator/val/test yang termasuk kata umum.
3. **Generator**: 15% alamat kini tanpa awalan (`cilandak raya jakarta selatan`,
   `diponegoro semarang`). Daerah di test_v4 sengaja tidak dimasukkan.
4. **Word vectors**: fastText `cc.id.300` (Common Crawl, CC BY-SA 3.0), 200 ribu kata teratas
   dipangkas menjadi 20 ribu vektor (`ner_service/prepare_vectors.py`), dipakai sebagai
   **fitur** tok2vec (`HashEmbedCNN.v2`, `pretrained_vectors=true`). Bobot NER tetap
   dilatih dari nol dengan data kita.
5. Pakai vektor atau tidak **diputuskan di val**, bukan di test: kedua varian dilatih dengan
   data yang sama.

| Varian (data v6 + filter) | val F2 terbaik | val F2 median 30 epoch |
|---|---|---|
| tanpa vektor | 0.89 | ~0.80 |
| **dengan vektor** | **0.96** (epoch 1) | **~0.91** |

Lalu test_v4 dibuka **sekali**:

| Model | test_v4 (buta) P / R / F2 | test_v4 bersih | test_v3 P / R / F2 | test_v3 bersih | guardrail e2e (29 PII) |
|---|---|---|---|---|---|
| v5 | 0.42 / 0.53 / 0.51 | 12/18 | 0.63 / 1.00 / 0.90 | 14/20 | 29 tertutup, 2 kata ikut tersensor |
| v5 + filter | 0.57 / 0.53 / 0.54 | 14/18 | 0.63 / 1.00 / 0.90 | 14/20 | — |
| v6 tanpa vektor + filter | 0.55 / 0.80 / 0.73 | 12/18 | 0.61 / 0.92 / 0.83 | 13/20 | — |
| **v6 dengan vektor + filter** | **0.92 / 0.80 / 0.82** | **18/18** | **1.00 / 1.00 / 1.00** | **20/20** | **29 tertutup, 0 kata ikut tersensor** |

**Temuan terpenting:** data tambahan saja menaikkan recall (alamat tanpa awalan mulai
tertangkap) tapi **menambah** salah sensor (bersih 14 → 12). Vektor yang membalik keadaan:
seluruh kalimat tanpa PII di test_v3 & test_v4 kini bersih. Artinya akar salah sensor
memang "model tidak tahu arti kata", bukan kekurangan data.

**Harga yang dibayar (jujur):**
- Model 4,7 MB → **34 MB**; RAM proses ~160 → **~250 MB**; waktu muat ~3,5 → ~4,9 detik.
  Request memori pod NER dinaikkan 256Mi → 384Mi. Latency tetap ~2–10 ms.
- Arti "buatan sendiri" direvisi: soal meminta *melatih model NER sendiri* dan eksplisit
  *boleh memakai library / framework open-source*. Bobot NER dilatih dari nol di data
  kita; vektor fastText hanya kamus fitur, bukan model NER orang lain. PRD §6 diperbarui.
- test_v4 dipakai sekali untuk baseline v5 **dan** filter pertama dirancang setelah melihat
  error baseline itu (mis. "apakah"). Dua kata yang diambil langsung dari error test_v4
  ("upgrade", "password") dibuang lagi dari daftar sebelum evaluasi akhir; sisanya
  kosakata CS umum. Tetap ada sedikit kebocoran test → angka filter agak optimistis.
  Model v6 sendiri dipilih murni dari val.

Sisa kesalahan di test_v4 (4): `ciputat tangerang selatan` **lolos utuh**, `agustinus lamere`
**lolos utuh** (nama di ujung kalimat setelah "pelanggan"), dan `fatmawati jakarta, apakah`
(batas kelebihan — alamat tetap tertutup). Di test_v2: `Jakarta Timur` ditebak PERSON
(tetap tersensor) dan `komang adi putra` hanya sebagian.

---

## v5 — salah sensor pada kalimat curhat (`pii_ner_id-2.1.0`)

**Pemicu (22 Sep, demo):** kalimat tanpa PII seperti "aku lagi pusing sama pola pikir orang"
disensor menjadi `aku [REDACT_NAMA] sama [REDACT_NAMA]`. Sebabnya: di data latih v4 **setiap**
"aku/saya + kata" diikuti nama, sehingga model yang tidak punya embedding (tidak tahu arti
kata) belajar "kata sesudah aku = nama". Kalimat demo itu sendiri tidak masuk data mana pun.

Urutan, mengikuti disiplin v4:

1. Tulis **`test_v3.jsonl`** (30 kalimat: 20 curhat/keluhan tanpa PII, termasuk nominal Rp/IDR,
   dan 10 kalimat bernama) **sebelum** generator diubah. Ukur v4 di situ: hanya **8/20**
   kalimat tanpa PII yang tetap bersih.
2. Tambah 12 kalimat ke **`val.jsonl`** (8 negatif sulit, 4 positif), karena val lama hanya
   punya 5 kalimat negatif sehingga pemilihan epoch nyaris tidak melihat salah sensor.
3. Generator: kalimat rakitan (subjek × keterangan × ~110 kata keadaan/aktivitas), kalimat
   bernominal rupiah, dan pola "curhat dulu, lalu nama". **Kata-kata test_v3 sengaja tidak
   dimasukkan** (dicek otomatis), jadi test_v3 mengukur generalisasi ke kata yang belum pernah
   dilihat. Train 2.700 → 3.600 kalimat, 51% tanpa entity.
4. Percobaan pertama: precision val naik ke 1.00, tetapi **nama huruf kecil mulai lolos**
   (`fauzan alfarizi`, `yohana siahaan`). Artinya model sebelumnya *menghafal* ~74 nama depan.
   Daftar nama diperluas ke ~200 nama depan & ~130 nama belakang lintas daerah; generator kini
   menolak jalan bila ada nama val/test di daftarnya. Terpilih **epoch 4**.

| Model | test_v3 (buta): kalimat bersih | test_v3 P / R / F2 | val: kalimat bersih | val P / R / F2 |
|---|---|---|---|---|
| v4 | 8/20 | 0.44 / 1.00 / 0.80 | 8/13 | 0.74 / 0.87 / 0.84 |
| **v5** | **14/20** | **0.63 / 1.00 / 0.90** | **12/13** | **0.95 / 0.87 / 0.88** |

End-to-end (regex + NER, 29 item PII): tetap **29 tertutup, 0 bocor**.

**Harga yang dibayar (jujur):** dihitung tanpa melihat label (di agent, PERSON yang ditebak
ADDRESS tetap tersensor), entity yang **lolos utuh** di semua set naik dari 3 (+2 sebagian)
menjadi 5 (+1 sebagian): `komang adi putra` (v4: sebagian), `mulyadi`, `yohanes ndruru`,
plus `Jl. Ir. H. Juanda …` yang hanya tertutup separuh. Dengan ~80 entity total, selisih ini
bisa noise, tetapi arahnya jelas: salah sensor turun banyak, kebocoran naik sedikit.
Diterima pemilik project dengan pertimbangan itu.

Sisa salah sensor di test_v3 (6): `kesel sama sikap`, `kurang paham maksud rincian`,
`langsung putus`, `males ribet`, `karena gajian telat` (ADDRESS), `ngerasa`, `pengen tahu promo`
— semuanya kata yang belum pernah dilihat model. Ini batas model tanpa embedding; obat yang
lebih tuntas adalah pretrained embedding bahasa Indonesia (misalnya fastText `cc.id.300`)
sebagai fitur tok2vec.

---

## v4 — metodologi diperbaiki (`pii_ner_id-2.0.0`)

Masalah v1–v3: test set yang sama dilihat di setiap iterasi, sehingga keputusan
perbaikan ikut dipengaruhi test (*test-set leakage*). Perbaikan v4, **dalam urutan ini**:

1. Tulis **`val.jsonl`** (20 kalimat tangan, baru) → khusus untuk memilih epoch.
2. Tulis **`test_v2.jsonl`** (20 kalimat tangan, baru) **sebelum** data latih diubah →
   dievaluasi **sekali** di akhir. Nama/alamat di val & test_v2 dikeluarkan dari daftar generator.
3. Baru setelah itu data latih diperkaya: 3.000 kalimat, nama lintas daerah (Batak, Bali,
   Tionghoa-Indonesia, Arab-Indonesia, "br" marga), awalan alamat Kampung/Kp./Desa/Dusun/
   Jalan Raya/Km, gaya chat informal, 30% huruf kecil, alamat yang langsung disambung keluhan.
4. Epoch dipilih dengan **F2 di val** (recall ditimbang 2×), bukan F1 di dev sintetis
   (yang selalu ~1.00). Terpilih **epoch 3** — latihan lebih lama membuat skor val naik-turun
   (model makin hafal template): contoh nyata *early stopping*.

| Model | test_v2 (buta) P / R / F1 / F2 | recall longgar | test_v1 F1 | val F1 |
|---|---|---|---|---|
| v3 | 0.75 / 0.63 / 0.69 / 0.65 | 0.79 | 0.86 | 0.76 |
| **v4** | **0.85 / 0.89 / 0.87 / 0.89** | **1.00** | 0.93 | 0.94 |

**Temuan terpenting:** v3 mendapat F1 0.86 di test_v1 tapi hanya **0.69 di test_v2 yang buta**.
Selisih 0.17 itu bukti terukur bahwa test_v1 sudah terkontaminasi dan skornya menipu.

Sisa kesalahan v4 di test_v2 (5): `komang adi putra` hanya "putra" yang tertangkap (nama Bali
huruf kecil — sebagian tertutup), `Jln Letjen Suprapto 17 Semarang` batas kelebihan sampai
"lampu LOS merah", dan "jelek" ditandai PERSON (FP).

**Temuan setelah rilis (22 Sep, dari test browser):** model menebak `Rp` (dari `Rp1508000`)
sebagai PERSON — data latih belum memuat nominal rupiah. Ditambal dengan aturan domain di
`ner_client.plausible_entities` (simbol mata uang bukan nama), **tanpa** melatih ulang agar
test_v2 tidak "diintip" lagi. Iterasi v5: tambahkan kalimat bernominal (Rp, IDR) ke generator.
Evaluasi end-to-end sempat tidak melihat kasus ini karena pengukur over-redaction hanya
menghitung kata >= 3 huruf — sekarang >= 2.

Catatan jujur yang tersisa: semua set (train, val, test) ditulis oleh pihak yang sama,
sehingga tetap ada bias gaya penulisan. Uji paling jujur adalah chat pelanggan asli
yang sudah dianonimkan — di luar jangkauan tugas ini.

---

## v1–v3

Skor di test_v1 (20 kalimat, 22 entity).

| Iterasi | Perubahan | Train | Precision | Recall | F1 | Recall longgar |
|---|---|---|---|---|---|---|
| v1 | awal: 28 template | 510 | 0.71 | 0.77 | 0.74 | 0.95 |
| v2 | +7 template negatif, setelah demo menemukan FP ("Oke" → PERSON, "yang mana ya" → ADDRESS) | 510 | 0.77 | 0.77 | 0.77 | 0.82 |
| v3 | data latih 600 → 1.500 kalimat | 1.275 | **0.90** | **0.82** | **0.86** | 0.91 |

Dev F1 = 1.00 di **semua** iterasi — dev berasal dari generator yang sama dengan train,
sehingga hanya membuktikan model hafal template. Yang jujur hanya skor test.

## Catatan kejujuran metodologi

- **Satu entity = 4,5%** di test set sekecil ini. Selisih antar-iterasi sebagian besar noise;
  angka di atas bukan bukti statistik bahwa v3 lebih baik, hanya indikasi.
- v1 → v2 memperbaiki precision tapi **menurunkan recall longgar** (0.95 → 0.82). Untuk guardrail
  itu kemunduran, karena recall = tidak bocor. Perbaikan yang dipilih bukan mengutak-atik
  template sampai test bagus, melainkan obat yang berlaku umum: **data latih lebih banyak**.
- Melihat skor test di tiap iterasi sudah merupakan kebocoran kecil (*test-set leakage*):
  pilihan kita ikut dipengaruhi test set. Dengan waktu lebih banyak, langkah yang benar:
  pisahkan **validation set tulisan tangan** untuk memilih model, dan sentuh test set
  **sekali** di akhir.

## Kelemahan v3 (sebagian besar diperbaiki di v4)

| Kasus | Jenis | Kemungkinan sebab |
|---|---|---|
| `kevin tanoto` | FN | nama huruf kecil + pola "aku … rumahnya di" tidak ada di train |
| `Stefanus Lumban Gaol` | batas meleset | nama marga 3 kata, pola "jadi {nama}" tidak ada di train |
| `Kampung Rawa Bebek RT 02 RW 07, …` | FN | awalan "Kampung" tidak ada di train |
| `Jl. Pattimura No 8 Ambon putus terus` | batas kelebihan | model tidak tahu di mana alamat berakhir bila langsung disambung keluhan |

Arah perbaikan: variasi template dari chat nyata yang sudah dianonimkan, nama daerah yang
lebih beragam (Batak, Bali, Tionghoa-Indonesia), awalan alamat (Kampung, Dusun, Desa),
atau pretrained embedding bahasa Indonesia.
