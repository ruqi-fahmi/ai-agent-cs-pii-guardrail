# PRD — AI Agent Customer Support dengan PII Guardrail

> **Status:** Tahap 1–5 selesai + penyempurnaan (plugin sensor sebelum simpan, NER v4 dengan test buta, evaluasi end-to-end, /metrics, CI, model card). Tahap 6: file Docker/k8s siap, deploy GKE tertahan verifikasi billing GCP.
> **Terakhir diperbarui:** 22 September 2026
> **Sumber tugas:** `TAKE-HOME TEST Basic AI Engineer Infomedia.pdf` (6 halaman)
> **Untuk siapa:** user (pengerja take-home test) dan AI mana pun yang melanjutkan
> project ini. Baca file ini dulu sebelum menyentuh kode.

---

## 1. Ringkasan

Membangun **AI Agent customer support** berbasis **Google ADK + Gemini** yang tidak
pernah mengirim data pribadi pelanggan ke LLM. Setiap pesan user melewati dua lapis
guardrail sebelum sampai ke model:

1. **Regex** → NIK, email, nomor telepon (pola yang strukturnya pasti)
2. **NER buatan sendiri** → nama orang & alamat (pola yang tidak bisa di-regex)

Model NER dilatih sendiri dan dijalankan sebagai **service REST terpisah**, dipanggil
agent lewat HTTP. Keduanya di-container-kan dan di-deploy ke **GKE**.

**Target estimasi pengerjaan:** 2–4 hari.

**Yang dinilai** (PDF bagian 8): pemahaman konsep agent & guardrail, konsep model
ML, AI/ML as a Service, alur end-to-end berjalan, integrasi antar komponen.
→ Artinya setiap keputusan di dokumen ini harus bisa dijelaskan **alasannya**,
bukan cuma ada kodenya.

---

## 2. Scope

### Masuk scope
- Agent CS teks-in/teks-out (Gemini ≥ 2.0) dengan `before_model_callback`
- Guardrail regex: NIK, email, telepon + redaksi
- Model NER (PERSON, ADDRESS) dilatih sendiri, dataset kecil sintetis (v6: + word vectors sebagai fitur)
- Set uji ≥ 10 kalimat + laporan akurasi (precision / recall / F1)
- NER Service REST terpisah, dipanggil agent via HTTP
- Pengukuran CPU, memory, latency NER
- Dockerfile ×2 + manifest Kubernetes + deploy ke GKE
- README lengkap

### Di luar scope
- Akurasi NER tinggi (PDF: "tidak dituntut")
- Knowledge base / RAG / integrasi sistem tiket sungguhan
- Redaksi output LLM (input sudah bersih, jadi LLM tak punya PII untuk dibocorkan)
- Autentikasi pengguna, UI kustom (pakai `adk web` bawaan untuk demo)

---

## 3. Arsitektur

```
┌──────────┐   pesan asli    ┌────────────────────────────────────────────┐
│   User   │ ──────────────► │ AI Agent (Google ADK)                      │
└──────────┘                 │                                            │
      ▲                      │  before_model_callback(llm_request)        │
      │                      │   ├─ 1. Regex   : NIK/email/telepon         │
      │                      │   │              → [REDACT_NIK] dst         │
      │                      │   ├─ 2. HTTP POST /ner ──────────────┐     │
      │                      │   │                                   ▼     │
      │                      │   │              ┌──────────────────────────┐
      │                      │   │              │ NER Service (FastAPI)    │
      │                      │   │              │ model spaCy buatan sendiri│
      │                      │   │ ◄── entities │ PERSON, ADDRESS          │
      │                      │   │              └──────────────────────────┘
      │                      │   └─ 3. ganti span → [REDACT_NAMA] /        │
      │                      │         [REDACT_ADDRESS]                   │
      │                      │                 │ teks bersih              │
      │                      │                 ▼                          │
      │     jawaban          │           Gemini API                       │
      └──────────────────────┤                                            │
                             └────────────────────────────────────────────┘
```

### Kenapa bentuknya begini (bahan menjelaskan saat review)

| Keputusan | Alasan |
|---|---|
| Guardrail di `before_model_callback`, bukan di tool | Tool dipanggil *atas kehendak LLM* — berarti PII sudah terlanjur terbaca LLM. Callback berjalan *sebelum* request dikirim, jadi LLM tak pernah melihat data asli. |
| Regex dulu, baru NER | Pola berstruktur pasti (16 digit, `@`) lebih murah & akurat dengan regex. Menjalankannya dulu juga berarti PII tersebut **tidak ikut terkirim** ke NER Service lewat jaringan. |
| NER sebagai service terpisah | Model punya siklus hidup sendiri (retrain, versi) dan profil resource beda dari agent. Bisa di-scale, di-update, dan dipakai sistem lain tanpa menyentuh agent — inti "ML as a Service". |
| Redaksi **semua** turn user, bukan hanya pesan terakhir | ADK menyusun ulang `llm_request.contents` dari riwayat sesi di tiap giliran. Kalau hanya pesan terbaru yang dibersihkan, PII dari giliran sebelumnya ikut terkirim lagi. |
| Fail-closed bila NER Service mati *(usulan, lihat §10)* | Untuk guardrail PII, bocor lebih mahal daripada gagal menjawab. Callback mengembalikan `LlmResponse` penolakan sopan sehingga Gemini tidak dipanggil sama sekali. |

---

## 4. Komponen A — AI Agent (Google ADK)

| # | Kebutuhan | Kriteria diterima |
|---|---|---|
| A1 | Agent CS sederhana dengan instruksi persona CS | Menjawab pertanyaan umum layanan dalam Bahasa Indonesia |
| A2 | Model Gemini ≥ 2.0 | Dibaca dari env `GEMINI_MODEL`, default `gemini-2.5-flash` (cek model yang tersedia saat implementasi) |
| A3 | Input & output teks | Bisa diuji via `adk web` dan `adk api_server` |
| A4 | Guardrail di `before_model_callback` | Log menunjukkan teks yang sampai ke LLM sudah ter-redaksi |

Instruksi agent juga memberi tahu LLM bahwa token `[REDACT_*]` adalah data yang
sengaja disembunyikan — agar LLM tidak bingung atau meminta ulang datanya secara
mentah, dan cukup merujuknya ("data yang Anda kirim sudah kami terima").

---

## 5. Komponen B — Guardrail Regex

### Pola

| Entity | Pola baseline PDF | Masalah | Pola dipakai |
|---|---|---|---|
| NIK | `^[0-9]{16}$` | Anchor `^$` → hanya cocok bila *seluruh pesan* adalah NIK; NIK di tengah kalimat lolos | `(?<!\d)\d{16}(?!\d)` |
| Email | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+.[a-zA-Z]{2,}` | Titik sebelum TLD tidak di-escape → cocok dengan karakter apa pun | `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` |
| Telepon | `(+62\|62\|0)8[1-9][0-9]{6,9}` | `+` tidak di-escape → Python `re.error: nothing to repeat` | `(?<!\d)(?:\+62\|62\|0)[\s-]?8[1-9](?:[\s-]?\d){6,10}(?!\d)` — juga menangkap format `0812-3456-7890` |

### Urutan eksekusi
NIK → email → telepon. NIK lebih dulu karena kode provinsi Kalimantan Tengah
berawalan `62`; lookaround `(?<!\d)…(?!\d)` + urutan ini mencegah potongan NIK
tertangkap sebagai nomor telepon.

### Redaksi
`[REDACT_NIK]`, `[REDACT_EMAIL]`, `[REDACT_PHONE]`.
Fungsi mengembalikan teks bersih + daftar temuan (`type`, `start`, `end`) —
**nilai aslinya tidak pernah di-log**.

### Keterbatasan (didokumentasikan, bukan diperbaiki)
NIK yang ditulis berspasi (`3201 1234 …`), telepon rumah/non-seluler, email yang
disamarkan (`budi [at] gmail`).

---

## 6. Komponen C — Guardrail NER (model buatan sendiri)

### Pilihan model
**spaCy, pipeline kosong `spacy.blank("id")` + komponen `ner`, dilatih dari nol.**

| Opsi | Ukuran | Latency CPU | Kenapa (tidak) dipilih |
|---|---|---|---|
| **spaCy blank + NER** | beberapa MB | ~ms | ✅ ringan, ringkas, konsep training jelas, cocok jadi service kecil |
| CRF (sklearn-crfsuite) | < 1 MB | ~ms | Bagus untuk belajar feature engineering, tapi lebih banyak kode manual |
| Fine-tune IndoBERT | **473 MB (terukur)** | **21,8 ms p50 (terukur)** | **Lebih akurat** (lihat revisi di bawah), tapi 12× lebih besar, 2,8× lebih boros RAM, ~5× lebih lambat |

"Buatan sendiri" terpenuhi karena tidak memakai bobot pretrained — model belajar
murni dari dataset kita.

**Revisi v6 (22 Sep):** tafsiran di atas lebih ketat dari soal. Soal meminta *"melatih
model NER sendiri"* dan eksplisit *"boleh menggunakan library / framework open-source"*.
Model tanpa embedding terbukti tidak bisa membedakan kata umum dari nama ("apakah status"
disensor sebagai nama), jadi v6 menambah word vectors fastText `cc.id.300` (dipangkas ke
20 ribu kata) sebagai **fitur beku**. Bobot NER tetap dilatih dari nol dengan data kita —
yang dipakai dari luar hanya kamus makna kata, bukan model NER orang lain. Fine-tune
IndoBERT tetap tidak dipilih saat itu (alasan di tabel). Bukti dan harganya: `docs/ner-iterasi.md`.

**Revisi v11 (23 Sep) — klaim soal IndoBERT diganti pengukuran:** baris "Fine-tune IndoBERT"
di tabel di atas semula berisi tebakan. IndoBERT kemudian benar-benar di-fine-tune dengan
data latih yang sama dan diuji pada test set yang sama, dinilai fungsi metrik yang sama.
Hasilnya: **IndoBERT lebih akurat** — recall longgar 1.00 di seluruh test set, termasuk yang
masih bocor pada model kecil. Harganya terukur juga: 473 MB vs 40 MB, ~690 MB vs ~250 MB RAM,
21,8 ms vs 2–13 ms.

Yang dirilis **tetap model spaCy dari nol**, karena soal meminta model *sederhana buatan
sendiri*, guardrail berjalan di setiap pesan (ukuran & latency = biaya operasi), hasil
end-to-end sudah 32/32 tertutup, dan bobot IndoBERT tidak muat di repo publik. **Keduanya
tetap tersedia**: NER Service punya dua backend dengan kontrak API sama
(`NER_BACKEND=spacy` default, `NER_BACKEND=indobert`), jadi pilihan ini bisa dibalik tanpa
menyentuh agent. Angka lengkap: `docs/ner-iterasi.md`.

### Label
`PERSON`, `ADDRESS` → diredaksi menjadi `[REDACT_NAMA]`, `[REDACT_ADDRESS]`
(mengikuti contoh PDF).

### Dataset
| Set | Isi | Jumlah | Tujuan |
|---|---|---|---|
| Train | Kalimat sintetis dari template CS × daftar nama × daftar alamat | ~150–300 | Melatih |
| Dev | Pecahan acak dari generator yang sama | ~15% | Memantau loss/overfit saat training |
| **Test** | **Ditulis manual**, template & nama **berbeda** dari train | **≥ 10** (target 20) | Mengukur akurasi secara jujur |

Test set sengaja memakai pola kalimat yang tak pernah dilihat model: kalau diambil
dari generator yang sama, skornya akan tinggi palsu karena model cuma menghafal
template. Ini poin konsep ML utama yang perlu bisa dijelaskan.

Contoh (PDF): *"Nama saya Budi Santoso dan tinggal di Jalan Sudirman Jakarta"*
→ `Budi Santoso` = PERSON, `Jalan Sudirman Jakarta` = ADDRESS.

### Evaluasi
Precision, recall, F1 per label (entity-level, exact match) + contoh kesalahan
(false positive / false negative) dengan analisis singkat. Hasil ditulis ke
`docs/ner-evaluation.md`.

Untuk guardrail, **recall lebih penting daripada precision**: nama yang lolos =
bocor; kata biasa yang ikut ter-redaksi = cuma sedikit mengganggu jawaban.

---

## 7. Komponen D — NER Service

FastAPI + uvicorn, model dimuat sekali saat startup.

### Kontrak API

`POST /ner`
```json
// request
{ "text": "Nama saya Budi Santoso, tinggal di Jalan Sudirman Jakarta" }

// response 200
{
  "entities": [
    { "label": "PERSON",  "text": "Budi Santoso",           "start": 10, "end": 22 },
    { "label": "ADDRESS", "text": "Jalan Sudirman Jakarta", "start": 38, "end": 60 }
  ],
  "model_version": "ner-id-v1",
  "latency_ms": 2.4
}
```

`GET /health` → `{ "status": "ok", "model_loaded": true }` — dipakai probe Kubernetes.

- Validasi input: `text` wajib, maksimal ~5.000 karakter (422 bila lebih)
- Service **tidak me-log isi teks** — hanya panjang teks, jumlah entity, latency
- Hanya diekspos di dalam cluster (ClusterIP), tidak ke internet

---

## 8. Komponen E — Integrasi

Alur satu giliran chat:

1. User mengirim pesan → ADK menyusun `llm_request` dari riwayat sesi
2. `before_model_callback` berjalan untuk setiap `Content` ber-role `user`:
   1. regex redaction
   2. `POST {NER_SERVICE_URL}/ner` (timeout 2 detik)
   3. ganti span PERSON/ADDRESS dari belakang ke depan (agar offset tak bergeser)
3. Teks bersih ditulis kembali ke `llm_request` → diteruskan ke Gemini
4. NER gagal / timeout → callback mengembalikan `LlmResponse` penolakan (fail-closed)

Konfigurasi via env: `GOOGLE_API_KEY`, `GEMINI_MODEL`, `NER_SERVICE_URL`,
`GUARDRAIL_FAIL_MODE` (`closed` | `open`).

Catatan jujur: riwayat sesi di memori ADK tetap menyimpan pesan asli — yang dijamin
bersih adalah **apa yang dikirim ke LLM**. Untuk produksi, redaksi juga perlu
dilakukan sebelum pesan disimpan ke session store.

---

## 9. Komponen F & G — Resource dan Deployment

### F. Pengukuran resource (`benchmark/bench_ner.py`)
| Metrik | Cara ukur |
|---|---|
| Latency model saja | 1.000× `nlp(text)` in-process → p50 / p95 / p99 |
| Latency end-to-end HTTP | 1.000× `POST /ner` → p50 / p95 / p99 |
| Memory | RSS proses sebelum & sesudah model dimuat (psutil) + `docker stats` |
| CPU | CPU% selama beban + `kubectl top pod` di cluster |

Hasil + spesifikasi mesin uji ditulis ke `docs/resource-performance.md`, lalu
dipakai untuk menentukan `resources.requests/limits` di manifest k8s.

### G. Deployment
- `ner_service/Dockerfile`, `agent/Dockerfile` (python-slim, non-root)
- Image di-push ke **Artifact Registry**
- Manifest di `k8s/`:
  - `ner-service`: Deployment + Service **ClusterIP** + readiness/liveness `/health`
  - `cs-agent`: Deployment + Service **LoadBalancer** (`adk api_server` / `adk web`)
  - `GOOGLE_API_KEY` sebagai **Secret**, bukan di image
- Agent memanggil NER via DNS internal: `http://ner-service.<namespace>.svc.cluster.local`
- Diuji dulu di Kubernetes lokal (Docker Desktop), baru ke GKE Autopilot

---

## 10. Keputusan terbuka

| # | Pertanyaan | Usulan default |
|---|---|---|
| K1 | Perilaku saat NER Service mati | **Fail-closed**, bisa diubah via env |
| K2 | Gemini API key | Google AI Studio (gratis) — user perlu membuat |
| K3 | GKE sungguhan atau manifest + panduan | Lihat catatan di bawah |
| K4 | Deadline pengumpulan | ? |
| K5 | Lokasi repo | GitHub akun user, repo publik bila diminta penilai |

**Catatan K3:** lapisan teks PDF menulis bagian G sebagai *"Bonus (Opsional) … jika
tidak deploy, jelaskan"*, tetapi gambar halaman 4 berjudul *"G. Proses Deployment"*
tanpa kata opsional — kemungkinan dokumen direvisi dan GKE menjadi wajib. GKE butuh
akun GCP dengan billing aktif, dan `gcloud` belum terpasang di laptop. Rencana aman:
manifest dibuat & diuji di Kubernetes lokal apa pun keputusannya, deploy GKE
sungguhan di akhir bila akun siap.

---

## 11. Struktur repo

```
AI_ML/
├── agent/
│   ├── cs_agent/
│   │   ├── __init__.py
│   │   ├── agent.py              # root_agent + instruksi CS
│   │   └── guardrails/
│   │       ├── regex_pii.py      # B
│   │       ├── ner_client.py     # HTTP client ke NER Service
│   │       └── callback.py       # before_model_callback (E)
│   ├── requirements.txt
│   └── Dockerfile
├── ner_service/
│   ├── app.py                    # FastAPI (D)
│   ├── data/
│   │   ├── generate_train.py     # dataset sintetis
│   │   └── test.jsonl            # ≥10 kalimat manual
│   ├── train.py                  # C
│   ├── evaluate.py               # P/R/F1
│   ├── model/                    # hasil training (dibuat ulang oleh train.py)
│   ├── requirements.txt
│   └── Dockerfile
├── benchmark/bench_ner.py        # F
├── k8s/                          # G
├── tests/                        # pytest: regex, callback, API
├── docs/
│   ├── PRD.md                    # dokumen ini
│   ├── ner-evaluation.md
│   └── resource-performance.md
└── README.md
```

---

## 12. Rencana kerja (sambil belajar)

| Tahap | Hasil | Konsep yang dipelajari |
|---|---|---|
| 1 | Regex guardrail + unit test | Regex, lookaround, kenapa urutan pola penting |
| 2 | Dataset + training + evaluasi NER | Train/dev/test split, overfitting, P/R/F1 |
| 3 | NER Service FastAPI | Model serving, load sekali saat startup, kontrak API |
| 4 | Agent ADK + callback + integrasi | Siklus hidup agent, callback, fail-closed |
| 5 | Benchmark resource | p50/p95, RSS, kenapa perlu warm-up |
| 6 | Docker + k8s lokal → GKE | Container, Deployment/Service/Secret, probe |
| 7 | README + latihan menjelaskan | Menyusun cerita end-to-end untuk sesi review (termasuk §14) |

Tahap 1–4 = syarat minimum berjalan end-to-end. Setiap tahap selesai dengan sesuatu
yang bisa dijalankan dan diuji, bukan setengah jadi.

### Metode belajar — target: paham & mahir, bukan cuma "kodenya jalan"

Setiap tahap melewati lima langkah. Tahap dianggap **selesai** hanya bila langkah 4–5
lulus, bukan saat kodenya sudah hijau.

1. **Konsep dulu** — penjelasan singkat + analogi ke hal yang sudah dikuasai user
   (FastAPI, VPS, systemd, SQL, dashboard penagihan)
2. **Bangun bareng** — kode ditulis per bagian kecil, tiap bagian dijelaskan
   *kenapa*-nya, bukan hanya *apa*-nya
3. **User menjalankan sendiri** — perintah disediakan, user yang mengeksekusi dan
   membaca output-nya
4. **Latihan ubah** — user mengubah sesuatu dan menebak hasilnya *sebelum*
   menjalankan (mis. "hapus lookaround, NIK mana yang sekarang lolos?")
5. **Jelaskan balik** — user menjelaskan tahap itu dengan kata-katanya sendiri,
   seolah menjawab penilai; kekeliruan diluruskan saat itu juga

Pertanyaan latihan & jawabannya dicatat terpisah di luar repo — sekaligus menjadi
bahan persiapan sesi review di akhir (§14).

---

## 13. Pemetaan ke checklist PDF

| Checklist PDF | Dipenuhi di |
|---|---|
| A. Agent ADK, Gemini ≥ 2.0, teks in/out, Before Model Callback | §4, `agent/` |
| B. Regex NIK / email / telepon + redaksi | §5, `regex_pii.py` |
| C. NER Nama & Alamat, dilatih sendiri | §6, `ner_service/train.py` |
| D. NER service terpisah, REST, dipanggil agent | §7, `ner_service/app.py` |
| E. HTTP call, hasil NER jadi guardrail, alur dijelaskan | §8, `callback.py`, README |
| F. CPU, memory, latency | §9, `docs/resource-performance.md` |
| G. Deploy agent & NER ke GKE | §9, `k8s/` |
| Output: repo, source agent, source NER, README | §11 |
| ≥ 10 data uji NER | §6, `data/test.jsonl` |

---

## 14. Pengalaman sebelumnya — bahan cerita saat review

User sudah pernah membangun versi kecil dari inti tugas ini di project
**Dashboard Ticketing Bola Gembira**, fitur interpretasi kata kunci keluhan:
`dashboardticketingbolagembira-main/backend_skeleton/backend/app/insights.py`.

### Yang sudah dikerjakan di sana
- Resume keluhan pelanggan dikirim ke **Gemini** untuk dirangkum per kata kunci
- **Sebelum** prompt dirakit, PII disensor dengan regex (`_samples()`, baris 97–109):
  | Pola | Diganti |
  |---|---|
  | `nama pelanggan <…>` s/d label field berikutnya | `[nama]` |
  | `msisdn <digit>` | `[msisdn]` |
  | angka 5+ digit | `xxxxx` |
  | email | `[email]` |
- Fallback antar-model + retry saat 429/5xx; temuan lapangan: `gemini-2.0-flash`
  sering kena kuota (429), `gemini-2.5-flash` lebih stabil (baris 22–23)
- Di frontend (`frontend/src/utils/mask.js`), nomor HP & angka 16 digit (NIK/ID)
  disensor sebagian untuk tampilan
- Klasifikasi data PII sudah didokumentasikan (`docs/02-Klasifikasi-Data.md`)

### Apa yang berubah di take-home test ini, dan kenapa

| | Bola Gembira | Take-home test |
|---|---|---|
| Cara memanggil Gemini | `urllib` langsung ke REST API | Framework agent (Google ADK) |
| Posisi sensor | Di dalam fungsi perakit prompt | `before_model_callback` — satu titik yang menjaga **setiap** request ke LLM |
| Deteksi nama | Regex berbasis **label** (`nama pelanggan …`) | **Model NER** |
| Deteksi alamat | Tidak ada | Model NER |
| Bentuk | Satu aplikasi | Agent + NER Service terpisah (ML as a Service) |

### Poin kunci untuk diceritakan
1. **Regex nama bisa jalan di Bola Gembira karena notes tiket berformat template.**
   Chat customer support adalah teks bebas: *"halo kak, aku Budi, rumahku di
   Jl. Melati 5"* tidak punya label yang bisa dijadikan patokan. Di sinilah regex
   buntu dan NER dibutuhkan: model belajar dari *konteks kalimat*, bukan dari label.
2. **Sensor agresif itu disengaja.** `\d{5,}` → `xxxxx` mengorbankan precision demi
   recall — prinsip yang sama dengan §6: untuk guardrail, yang lolos (bocor) lebih
   mahal daripada yang kebanyakan disensor.
3. **Posisi sensor menentukan jaminannya.** Di Bola Gembira, sensor melekat pada satu
   fungsi; jalur kode lain yang memanggil Gemini bisa lupa menyensor. Callback ADK
   memusatkan guardrail di satu titik yang pasti dilewati semua request.
4. **Pengalaman operasional** — kuota 429, fallback model, retry — adalah hal nyata
   yang juga akan muncul di agent ini dan perlu ditangani.
