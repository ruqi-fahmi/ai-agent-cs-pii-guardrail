"""
Jalankan banyak konfigurasi train.py sekaligus (paralel), lalu ringkas skornya di val.

    python ner_service/sweep.py --workers 16 --out sweep_out

Tiap konfigurasi dilatih dengan beberapa seed: val kecil (62 kalimat) dan skornya naik-turun
antar-epoch, jadi satu run saja bisa menang karena untung. Konfigurasi dipilih dari
RATA-RATA seed, bukan run terbaik. test set tidak disentuh di sini.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".cache"

CONFIGS = {
    "A_v7_lower20k":          ["--data", CACHE / "data_v7", "--vectors", CACHE / "vec_lower20k"],
    "B_v6data_lower20k":      ["--data", CACHE / "data_v6", "--vectors", CACHE / "vec_lower20k"],
    "C_v7_orth20k":           ["--data", CACHE / "data_v7", "--vectors", CACHE / "vectors_id"],
    "D_v7_lower50k":          ["--data", CACHE / "data_v7", "--vectors", CACHE / "vec_lower50k"],
    "E_v7_lower20k_wide":     ["--data", CACHE / "data_v7", "--vectors", CACHE / "vec_lower20k",
                               "--width", "128", "--embed-size", "5000"],
    "F_v7_lower20k_drop02":   ["--data", CACHE / "data_v7", "--vectors", CACHE / "vec_lower20k",
                               "--dropout", "0.2"],
    "G_v7_lower50k_wide":     ["--data", CACHE / "data_v7", "--vectors", CACHE / "vec_lower50k",
                               "--width", "128", "--embed-size", "5000"],
    "H_v7_novec":             ["--data", CACHE / "data_v7", "--no-vectors"],
    # ronde 2: apakah E menang karena data v7 atau karena jaringan lebar?
    "I_v6data_lower20k_wide": ["--data", CACHE / "data_v6", "--vectors", CACHE / "vec_lower20k",
                               "--width", "128", "--embed-size", "5000"],
    "J_v7_orth20k_wide":      ["--data", CACHE / "data_v7", "--vectors", CACHE / "vectors_id",
                               "--width", "128", "--embed-size", "5000"],
    # v8: data + hari/bulan, dua varian vektor terbaik dari sweep v7
    "K_v8_lower20k_wide":     ["--data", CACHE / "data_v8", "--vectors", CACHE / "vec_lower20k",
                               "--width", "128", "--embed-size", "5000"],
    "L_v8_orth20k_wide":      ["--data", CACHE / "data_v8", "--vectors", CACHE / "vectors_id",
                               "--width", "128", "--embed-size", "5000"],
    # v9: alamat tanpa awalan diperbanyak & diperkaya
    "M_v9_orth20k_wide":      ["--data", CACHE / "data_v9", "--vectors", CACHE / "vectors_id",
                               "--width", "128", "--embed-size", "5000"],
    "N_v9_lower20k_wide":     ["--data", CACHE / "data_v9", "--vectors", CACHE / "vec_lower20k",
                               "--width", "128", "--embed-size", "5000"],
    # v10: batas alamat (arah mata angin) + frasa lokasi umum sebagai kalimat negatif
    "O_v10_orth20k_wide":     ["--data", CACHE / "data_v10", "--vectors", CACHE / "vectors_id",
                               "--width", "128", "--embed-size", "5000"],
    "P_v10_lower20k_wide":    ["--data", CACHE / "data_v10", "--vectors", CACHE / "vec_lower20k",
                               "--width", "128", "--embed-size", "5000"],
}


def run(name: str, args: list, seed: int, epochs: int, out: Path) -> dict:
    run_dir = out / f"{name}_s{seed}"
    log_file = out / f"{name}_s{seed}.log"
    cmd = [sys.executable, str(HERE / "train.py"), *map(str, args), "--seed", str(seed),
           "--epochs", str(epochs), "--out", str(run_dir)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1"}           # 1 core per run -> tidak saling rebut
    with open(log_file, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
    if rc != 0:
        return {"name": name, "seed": seed, "error": f"exit {rc}, lihat {log_file}"}
    log = json.loads((run_dir / "training_log.json").read_text(encoding="utf-8"))
    return {"name": name, "seed": seed, **log["best_epoch"], "train_seconds": log["train_seconds"],
            "median_val_f2": statistics.median(e["val_f2"] for e in log["epochs"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0, help="seed pertama (ronde baru = seed baru)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--only", nargs="*", help="nama konfigurasi (default: semua)")
    ap.add_argument("--out", default=str(CACHE / "sweep"))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(n, a, s) for n, a in CONFIGS.items() if not args.only or n in args.only
            for s in range(args.seed_start, args.seed_start + args.seeds)]
    print(f"{len(jobs)} run, {args.workers} paralel", flush=True)
    results = []
    with ThreadPoolExecutor(args.workers) as pool:
        for r in pool.map(lambda j: run(*j, args.epochs, out), jobs):
            results.append(r)
            print(json.dumps(r), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'konfigurasi':24} {'F2 rata2':>9} {'F2 min':>7} {'median/epoch':>13} {'bersih':>8} {'epoch':>6}")
    by = {}
    for r in results:
        if "error" not in r:
            by.setdefault(r["name"], []).append(r)
    for name, rs in sorted(by.items(), key=lambda kv: -statistics.mean(r["val_f2"] for r in kv[1])):
        print(f"{name:24} {statistics.mean(r['val_f2'] for r in rs):9.3f} "
              f"{min(r['val_f2'] for r in rs):7.3f} "
              f"{statistics.mean(r['median_val_f2'] for r in rs):13.3f} "
              f"{statistics.mean(r['val_clean'] for r in rs):5.1f}/{rs[0]['val_neg']} "
              f"{'/'.join(str(r['epoch']) for r in rs):>6}")
    for r in results:
        if "error" in r:
            print("GAGAL:", r)


if __name__ == "__main__":
    main()
