# Alur Journey — dari pesan pelanggan sampai deploy

Dokumen ini menjelaskan sistem dari sudut pandang **perjalanan**, bukan daftar fitur.
Tiga journey:

- **A. Journey satu pesan** — apa yang terjadi antara pelanggan menekan Enter dan jawaban muncul
- **B. Journey model NER** — kenapa modelnya sampai versi 3.x, apa yang diperbaiki tiap iterasi
- **C. Journey deployment** — dari kode di laptop sampai jalan di Google Kubernetes Engine

Peta singkat siapa berbicara ke siapa:

```
   Pelanggan
      │  teks asli (boleh berisi NIK, email, telepon, nama, alamat)
      ▼
┌──────────────────────────── AI Agent (Google ADK) ────────────────────────────┐
│  Plugin  ──► Regex ──► NER Client ──HTTP──► [ NER Service ]  ──► Redaksi      │
│                                              FastAPI + spaCy                  │
│                                   teks bersih ──► Gemini ──► rapikan jawaban  │
└───────────────────────────────────────────────────────────────────────────────┘
      │  jawaban
      ▼
   Pelanggan
```

Dua kotak itu = dua container = dua Deployment di GKE. Yang menyimpan model hanya NER Service;
yang memanggil Gemini hanya Agent.

---

## A. Journey satu pesan

Contoh nyata (diambil dari demo di GKE):

> `coba cek nomor ini 6281260560185 terdaftar atas nama fahmi dengan alamat di fatmawati jakarta, apakah status masih aktif?`

### Langkah 1 — Pesan masuk lewat HTTP
`POST /api/chat` di [`agent/server.py`](../agent/server.py) menerima `{session_id, message}`.
Teks masih **utuh** di sini: ini proses milik kita sendiri, belum keluar ke pihak ketiga.

### Langkah 2 — Plugin menyensor SEBELUM disimpan ke riwayat sesi
[`PiiRedactionPlugin.on_user_message_callback`](../agent/cs_agent/guardrails/callback.py)

Kenapa di sini duluan: ADK menyimpan tiap pesan user ke *session state*. Kalau penyensoran
baru terjadi menjelang LLM, PII mentah sudah terlanjur mengendap di memori server (dan ikut
terbawa di giliran berikutnya). Jadi pintu pertama adalah pintu **penyimpanan**.

### Langkah 3 — Guardrail lapis 1: regex
[`regex_pii.py`](../agent/cs_agent/guardrails/regex_pii.py) — NIK, email, telepon.
Urutannya **email dulu, baru telepon**, supaya `081234567890@gmail.com` tertangkap utuh
sebagai email, bukan terpotong jadi nomor telepon.

Hasil di contoh: `6281260560185` → `[REDACT_PHONE]`.

### Langkah 4 — Guardrail lapis 2: NER lewat HTTP
[`ner_client.py`](../agent/cs_agent/guardrails/ner_client.py) memanggil
`POST http://ner-service/ner` (timeout 2 detik) dengan teks yang **sudah** bebas NIK/email/telepon.
Nomor sensitif tidak pernah melintas jaringan ke service lain.

Di dalam [`ner_service/app.py`](../ner_service/app.py): model spaCy menebak span, lalu
[`postprocess.py`](../ner_service/postprocess.py) memangkas kata umum di tepi tebakan nama
("apakah status" dibuang, "untuk pelanggan agustinus" dipangkas jadi "agustinus").
Yang dikembalikan hanya label + posisi (`start`, `end`), bukan penilaian apa pun soal pelanggan.

Hasil di contoh: `fahmi` → PERSON, `fatmawati jakarta` → ADDRESS.

### Langkah 5 — Redaksi
Span diganti dari **belakang ke depan** — kalau dari depan, panjang teks berubah dan offset
berikutnya meleset. Yang sampai ke Gemini:

```
coba cek nomor ini [REDACT_PHONE] terdaftar atas nama [REDACT_NAMA] dengan alamat di [REDACT_ADDRESS] status masih aktif?
```

### Langkah 6 — Pintu kedua: `before_model_callback`
Teks yang sama diperiksa **lagi** tepat sebelum dikirim ke LLM. Ini bukan pemborosan, tapi
*defense in depth*: ia menjaga jalur yang tidak lewat plugin (mis. runner lain, atau pesan
yang disuntikkan langsung ke sesi). Hasil per pesan di-cache dengan kunci hash, jadi teks
yang sudah bersih tidak dikirim ulang ke NER.

### Langkah 7 — Gemini menjawab
Model utama `gemini-2.5-flash`; kalau kuota habis (429/503/404), [`FallbackGemini`](../agent/cs_agent/model.py)
otomatis pindah ke model cadangan. LLM diberi tahu lewat system instruction bahwa token
`[REDACT_*]` berarti "pelanggan sudah memberikan data itu, tapi disembunyikan sistem" — supaya
ia tidak menanyakan ulang data yang sudah dikirim pelanggan.

### Langkah 8 — Pintu keluar: `after_model_callback`
Kalau LLM membeo token mentah (`[REDACT_ADDRESS]`) di jawabannya, token itu diganti kalimat
wajar: "alamat yang Bapak/Ibu berikan".

### Langkah 9 — Jawaban + panel bukti
Halaman demo menampilkan: PII yang terdeteksi (diwarnai per lapis), **teks persis yang dikirim
ke Gemini**, bukti bahwa isi riwayat sesi sama dengan itu, model yang menjawab, dan waktu tiap
tahap. Server hanya mengirim posisi PII, tidak pernah nilainya; pewarnaan dikerjakan browser.

### Kalau NER Service mati?
**Fail-closed**: plugin menyimpan `[PESAN DITAHAN…]` dan callback mengembalikan penolakan —
**Gemini tidak dipanggil sama sekali**. Lebih baik pelanggan menunggu daripada PII bocor.
Bisa diubah ke `GUARDRAIL_FAIL_MODE=open` (regex saja) kalau ketersediaan lebih diutamakan.

### Waktu yang dihabiskan (terukur di GKE)
| Tahap | Waktu |
|---|---|
| Regex | 0,05 ms |
| NER (termasuk HTTP antar-pod) | 14–43 ms |
| Gemini | 2,5–4 detik |
| **Total** | **2,5–5 detik** |

Guardrail bukan bottleneck: hampir seluruh waktu tunggu adalah LLM.

---

## B. Journey model NER

Satu pola berulang di tiap versi: **demo menemukan kegagalan → tulis test set baru & kunci →
perbaiki → ukur sekali**. Rinciannya di [ner-iterasi.md](ner-iterasi.md).

| Versi | Masalah yang memicu | Yang diubah | Hasil |
|---|---|---|---|
| v1–v3 | — | perbanyak data latih | F1 0.74 → 0.86 di test_v1 |
| **v4** | skor test_v1 ternyata menipu | pisahkan **val** (pilih model) & **test_v2** (dikunci, dibuka sekali) | v3 dapat 0.86 di test lama tapi **0.69** di test buta — bukti *test-set leakage* |
| **v5** | demo: "aku lagi pusing sama pola pikir orang" tersensor | kalimat curhat & nominal rupiah di data latih | kalimat tanpa PII yang bersih 8/20 → 14/20 |
| **v6** | demo GKE: `apakah status` tersensor, `fatmawati jakarta` **bocor** | word vectors fastText sebagai fitur + pangkas kata umum + alamat tanpa awalan | test_v4: P 0.42 → 0.92; bersih 18/18 |
| **v7** | masih ada 2 kebocoran & batas alamat kelebihan | sweep 10 konfigurasi × 3–6 seed di VM GCP; val diperbesar 32 → 62 | **0 entity bocor utuh di 5 test set** (v6: 7); harga: nama hari/bulan ikut tersensor |
| **v8** | nama hari/bulan (`Sabtu`, `Agustus`) ditandai nama orang | hari & bulan masuk val + data latih; sweep 8 seed | salah sensor hilang: **20/20 kalimat bersih** di test_v6 & test_v5; harga: beberapa alamat tanpa awalan lolos |

Tiga aturan main yang dipegang sepanjang journey ini:

1. **Test set ditulis dan dikunci sebelum perubahan dibuat**, lalu dibuka sekali di akhir.
2. **Model dipilih di val, bukan di test.** Di v7 ada seed yang skornya sempurna di test —
   sengaja tidak dipakai, karena memilih lewat test membuat angkanya palsu.
3. **Metriknya F2** (recall ditimbang 2×): nama yang lolos = data bocor ke pihak ketiga,
   jauh lebih mahal daripada kata biasa yang ikut tersensor.

---

## C. Journey deployment

```
kode + model   ──gcloud builds submit──►  Artifact Registry  ──kubectl apply──►  GKE Autopilot
(laptop)            (build di server Google)   (image 1.x.x)        (2 Deployment, 1 namespace)
```

1. **Bangun image di server Google** — `gcloud builds submit`. Laptop hanya mengunggah kode
   beberapa MB; Docker lokal tidak diperlukan, dan hasilnya pasti Linux amd64 seperti node GKE.
2. **Cluster Autopilot** — Google yang mengelola node; kita hanya membayar resource yang
   di-*request* pod. Request diambil dari pengukuran nyata di
   [resource-performance.md](resource-performance.md), bukan tebakan.
3. **Model ikut di dalam image.** Kode dan bobot model terkunci bersama dalam satu tag versi,
   jadi tidak ada kejadian "kode baru, model lama".
4. **API key lewat Secret**, tidak pernah ikut ter-*bake* ke image atau ter-commit ke git.
5. **Dua Service ClusterIP.** NER Service menerima teks ber-PII → tidak boleh dijangkau dari
   internet. Halaman demo agent juga tidak dibuka publik (tanpa autentikasi + API key berbayar);
   aksesnya lewat `kubectl port-forward`.
6. **NER 2 replika, Agent 1 replika.** NER harus tahan satu pod mati karena guardrail-nya
   fail-closed; agent menyimpan sesi di memori pod, jadi >1 replika butuh session store bersama.
7. **Metrik** `/metrics` di-scrape Managed Prometheus — isinya hanya angka (jumlah request,
   latency, jumlah entity), tidak pernah isi teks.
8. **Bersih-bersih**: `gcloud container clusters delete` supaya kredit tidak habis saat menganggur.

Langkah perintahnya ada di [DEPLOY-GKE.md](DEPLOY-GKE.md).

---

## Peta file — kalau ingin membaca kodenya

| Pertanyaan | File |
|---|---|
| Di mana guardrail dipasang ke ADK? | [`agent/cs_agent/guardrails/callback.py`](../agent/cs_agent/guardrails/callback.py) |
| Pola regex PII? | [`agent/cs_agent/guardrails/regex_pii.py`](../agent/cs_agent/guardrails/regex_pii.py) |
| Cara agent memanggil NER? | [`agent/cs_agent/guardrails/ner_client.py`](../agent/cs_agent/guardrails/ner_client.py) |
| API NER Service? | [`ner_service/app.py`](../ner_service/app.py) |
| Cara model dilatih & dipilih? | [`ner_service/train.py`](../ner_service/train.py), [`sweep.py`](../ner_service/sweep.py) |
| Dari mana data latih? | [`ner_service/data/generate_train.py`](../ner_service/data/generate_train.py) |
| Angka akurasi? | [ner-evaluation.md](ner-evaluation.md), [MODEL_CARD.md](MODEL_CARD.md) |
| Berapa PII yang bocor end-to-end? | [guardrail-evaluation.md](guardrail-evaluation.md) |
| CPU/memory/latency? | [resource-performance.md](resource-performance.md) |
| Kenapa keputusannya begitu? | [PRD.md](PRD.md), [ner-iterasi.md](ner-iterasi.md) |

---

## Pertanyaan yang mungkin ditanyakan penilai

| Pertanyaan | Jawaban singkat | Bukti |
|---|---|---|
| Kenapa guardrail di callback, bukan di tool? | Tool dipanggil atas kehendak LLM — artinya LLM sudah membaca PII-nya. Callback berjalan sebelum request dikirim. | README "Kenapa bentuknya begini" |
| Kenapa NER dipisah jadi service? | Siklus rilis & profil resource beda dari agent; bisa di-scale/retrain sendiri dan dipakai sistem lain lewat kontrak API yang sama. | PRD §7 |
| Modelnya benar-benar dilatih sendiri? | Ya — bobot NER dari nol dengan dataset kita. Word vectors fastText hanya **fitur beku** (kamus makna kata), bukan model NER orang lain. | MODEL_CARD, PRD §6 revisi v6 |
| Kenapa akurasinya tidak 100%? | Soal tidak menuntut akurasi tinggi; yang dijaga adalah **kebocoran**. Recall longgar 1.00 = tidak ada nama/alamat yang lolos utuh di semua test set. | ner-evaluation.md |
| Bagaimana tahu skornya tidak menipu? | Test set dikunci sebelum perubahan, dibuka sekali; model dipilih di val. Bukti nyata bahaya leakage: v3 dapat 0.86 di test lama, 0.69 di test buta. | ner-iterasi.md v4 |
| Kalau NER Service mati? | Fail-closed: Gemini tidak dipanggil, pesan ditahan. | callback.py, tests/test_callback.py |
| PII bisa bocor lewat log? | Log & metrik hanya angka (panjang teks, jumlah entity, latency); ada test yang memastikan isi teks tidak muncul di `/metrics`. | tests/test_ner_service.py |
