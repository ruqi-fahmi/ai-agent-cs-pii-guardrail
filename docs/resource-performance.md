# Resource & Performance — NER Service

> Ditulis ulang otomatis oleh `python benchmark/bench_ner.py --url http://127.0.0.1:8020`.

## Mesin uji

- CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel — 12 core fisik / 16 thread
- RAM: 15.7 GB
- OS: Windows 11, Python 3.12.10
- Jumlah request per skenario: 1000 (setelah 20 request warm-up)

## Memory

| Komponen | RSS |
|---|---|
| Proses Python kosong | 20 MB |
| + library spaCy | 80 MB (+60) |
| + model NER dimuat | 233 MB (+153) |
| Proses NER Service (uvicorn + FastAPI + model), setelah beban | 249 MB |

Ukuran model di disk: **37.3 MB**. Waktu muat model: **4710 ms** (sekali saat service start).

## Latency & CPU

| Jalur | Panjang teks | p50 | p95 | p99 | Throughput (1 proses) | CPU terpakai |
|---|---|---|---|---|---|---|
| model in-process | pendek (~60 karakter) | 2.09 ms | 2.98 ms | 3.33 ms | 456 req/s | 0.98 core |
| model in-process | sedang (~200 karakter) | 3.86 ms | 5.18 ms | 5.93 ms | 250 req/s | 0.99 core |
| model in-process | panjang (~1.000 karakter) | 12.66 ms | 16.16 ms | 21.33 ms | 77 req/s | 0.98 core |
| HTTP POST /ner | pendek (~60 karakter) | 4.74 ms | 5.76 ms | 6.39 ms | 207 req/s | — |
| HTTP POST /ner | sedang (~200 karakter) | 6.46 ms | 7.78 ms | 8.92 ms | 152 req/s | — |
| HTTP POST /ner | panjang (~1.000 karakter) | 14.09 ms | 16.05 ms | 16.88 ms | 70 req/s | — |

## Cara membaca

- **p50** = median: setengah request lebih cepat dari angka ini. **p95/p99** = ekor lambat: 5% / 1% request lebih lambat dari ini — yang dirasakan pengguna saat sial.
- **CPU terpakai ≈ 1 core** saat beban penuh berarti inferensi berjalan di satu thread; throughput naik dengan menambah proses/replika, bukan thread.
- Dibanding panggilan Gemini (~1,5–4 detik per giliran pada demo), overhead guardrail NER hanya milidetik — tidak menjadi bottleneck.

## Rekomendasi resource Kubernetes

Dipakai di `k8s/ner-service.yaml`:

| | CPU | Memory | Alasan |
|---|---|---|---|
| requests | 250m | 256Mi | beban CS normal jauh di bawah 1 core; RSS terukur di atas + ruang |
| limits | 1 | 512Mi | inferensi single-thread → >1 core tidak berguna per pod; batas memori ~2× RSS untuk cegah OOM saat teks panjang |

Skala horizontal (lebih banyak replika) dipakai bila throughput 1 pod tidak cukup.
