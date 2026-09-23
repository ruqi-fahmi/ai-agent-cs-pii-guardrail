"""
Ukur resource & latency model NER (poin F di soal).

    python benchmark/bench_ner.py                       # model in-process saja
    python benchmark/bench_ner.py --url http://127.0.0.1:8020   # + lewat HTTP

Yang diukur:
  - Memory  : RSS (RAM fisik yang dipakai proses) sebelum & sesudah model dimuat
  - Latency : waktu nlp(text) — p50/p95/p99 setelah warm-up
              (request pertama selalu lebih lambat: alokasi memori, cache CPU dingin)
  - CPU     : waktu CPU / waktu dinding selama beban -> berapa core yang terpakai
  - HTTP    : latency end-to-end POST /ner (model + serialisasi JSON + jaringan lokal)

Hasil ditulis ke docs/resource-performance.md.
"""
import argparse
import os
import platform
import statistics
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ner_service" / "model"
REPORT = ROOT / "docs" / "resource-performance.md"

TEXTS = {
    "pendek (~60 karakter)": "Nama saya Budi Santoso dan tinggal di Jalan Sudirman Jakarta",
    "sedang (~200 karakter)": (
        "Halo kak, saya Rahmat Syahputra. Internet di rumah saya Jl. Pattimura No 8 Ambon "
        "mati sejak kemarin sore, sudah restart modem tapi lampu LOS masih merah. Mohon "
        "dikirim teknisi secepatnya ya kak."),
}
TEXTS["panjang (~1.000 karakter)"] = " ".join([TEXTS["sedang (~200 karakter)"]] * 5)


def pct(values, p):
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(p / 100 * (len(values) - 1))))
    return values[k]


def bench(fn, n, warmup=20):
    for _ in range(warmup):
        fn()
    proc = psutil.Process()
    cpu0, wall0 = proc.cpu_times(), time.perf_counter()
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000)
    wall = time.perf_counter() - wall0
    cpu1 = proc.cpu_times()
    cpu_s = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    return {"p50": statistics.median(lat), "p95": pct(lat, 95), "p99": pct(lat, 99),
            "mean": statistics.fmean(lat), "rps": n / wall, "cores": cpu_s / wall}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--url", help="URL NER Service untuk uji HTTP, mis. http://127.0.0.1:8020")
    args = ap.parse_args()

    proc = psutil.Process()
    rss_base = proc.memory_info().rss
    import spacy   # import di sini supaya memori library ikut terhitung terpisah
    rss_lib = proc.memory_info().rss
    t0 = time.perf_counter()
    nlp = spacy.load(MODEL_DIR)
    load_ms = (time.perf_counter() - t0) * 1000
    rss_model = proc.memory_info().rss

    rows = []
    for name, text in TEXTS.items():
        r = bench(lambda: nlp(text), args.n)
        rows.append(("model in-process", name, r))
        print(f"model  {name:24} p50={r['p50']:.2f}ms p95={r['p95']:.2f}ms "
              f"p99={r['p99']:.2f}ms  {r['rps']:.0f} req/s  CPU={r['cores']:.2f} core")

    if args.url:
        import httpx
        with httpx.Client(base_url=args.url, timeout=10) as client:
            for name, text in TEXTS.items():
                r = bench(lambda: client.post("/ner", json={"text": text}).raise_for_status(), args.n)
                rows.append(("HTTP POST /ner", name, r))
                print(f"http   {name:24} p50={r['p50']:.2f}ms p95={r['p95']:.2f}ms "
                      f"p99={r['p99']:.2f}ms  {r['rps']:.0f} req/s")
        # Di Windows, python.exe di venv hanya peluncur yang menjalankan interpreter
        # asli sebagai proses anak — ambil RSS terbesar di antara proses + anaknya.
        server_rss = None
        for p in psutil.process_iter(["cmdline"]):
            cmd = " ".join(p.info["cmdline"] or [])
            if "uvicorn" in cmd and "app:app" in cmd and "ner_service" in cmd:
                for q in [p, *p.children(recursive=True)]:
                    rss = q.memory_info().rss / 2**20
                    server_rss = max(server_rss or 0, rss)
    else:
        server_rss = None

    mb = lambda b: b / 2**20
    model_size = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file())
    cpu_name = platform.processor() or platform.machine()
    lines = [
        "# Resource & Performance — NER Service",
        "",
        "> Ditulis ulang otomatis oleh `python benchmark/bench_ner.py"
        + (f" --url {args.url}" if args.url else "") + "`.",
        "",
        "## Mesin uji",
        "",
        f"- CPU: {cpu_name} — {psutil.cpu_count(logical=False)} core fisik / "
        f"{psutil.cpu_count()} thread",
        f"- RAM: {mb(psutil.virtual_memory().total) / 1024:.1f} GB",
        f"- OS: {platform.system()} {platform.release()}, Python {platform.python_version()}",
        f"- Jumlah request per skenario: {args.n} (setelah 20 request warm-up)",
        "",
        "## Memory",
        "",
        "| Komponen | RSS |",
        "|---|---|",
        f"| Proses Python kosong | {mb(rss_base):.0f} MB |",
        f"| + library spaCy | {mb(rss_lib):.0f} MB (+{mb(rss_lib - rss_base):.0f}) |",
        f"| + model NER dimuat | {mb(rss_model):.0f} MB (+{mb(rss_model - rss_lib):.0f}) |",
    ]
    if server_rss:
        lines.append(f"| Proses NER Service (uvicorn + FastAPI + model), setelah beban | {server_rss:.0f} MB |")
    lines += [
        "",
        f"Ukuran model di disk: **{mb(model_size):.1f} MB**. Waktu muat model: **{load_ms:.0f} ms** "
        "(sekali saat service start).",
        "",
        "## Latency & CPU",
        "",
        "| Jalur | Panjang teks | p50 | p95 | p99 | Throughput (1 proses) | CPU terpakai |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, name, r in rows:
        cpu = f"{r['cores']:.2f} core" if kind.startswith("model") else "—"
        lines.append(f"| {kind} | {name} | {r['p50']:.2f} ms | {r['p95']:.2f} ms | "
                     f"{r['p99']:.2f} ms | {r['rps']:.0f} req/s | {cpu} |")
    lines += [
        "",
        "## Cara membaca",
        "",
        "- **p50** = median: setengah request lebih cepat dari angka ini. **p95/p99** = ekor "
        "lambat: 5% / 1% request lebih lambat dari ini — yang dirasakan pengguna saat sial.",
        "- **CPU terpakai ≈ 1 core** saat beban penuh berarti inferensi berjalan di satu thread; "
        "throughput naik dengan menambah proses/replika, bukan thread.",
        "- Dibanding panggilan Gemini (~1,5–4 detik per giliran pada demo), overhead guardrail "
        "NER hanya milidetik — tidak menjadi bottleneck.",
        "",
        "## Rekomendasi resource Kubernetes",
        "",
        "Dipakai di `k8s/ner-service.yaml`:",
        "",
        "| | CPU | Memory | Alasan |",
        "|---|---|---|---|",
        "| requests | 250m | 256Mi | beban CS normal jauh di bawah 1 core; RSS terukur di atas + ruang |",
        "| limits | 1 | 512Mi | inferensi single-thread → >1 core tidak berguna per pod; "
        "batas memori ~2× RSS untuk cegah OOM saat teks panjang |",
        "",
        "Skala horizontal (lebih banyak replika) dipakai bila throughput 1 pod tidak cukup.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nmemori: base={mb(rss_base):.0f}MB +spacy={mb(rss_lib):.0f}MB +model={mb(rss_model):.0f}MB"
          f"  server={server_rss and round(server_rss)}MB  load={load_ms:.0f}ms")
    print(f"laporan: {REPORT}")


if __name__ == "__main__":
    main()
