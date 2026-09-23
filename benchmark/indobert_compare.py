"""
Pembanding: fine-tune IndoBERT untuk tugas yang sama, diukur dengan metrik yang sama.

    python benchmark/indobert_compare.py --epochs 3 --out .cache/indobert

Kenapa ada: PRD §6 menolak IndoBERT dengan alasan "berlebihan untuk tugas yang tidak
menuntut akurasi tinggi; image & RAM besar". Itu klaim, bukan pengukuran. Skrip ini
mengubahnya menjadi angka: model yang sama-sama dilatih dengan data kita, diuji pada
test set yang sama, dinilai `ner_service/evaluate.score_spans` yang sama.

Yang dibandingkan: akurasi per test set, ukuran di disk, RAM, dan latency inferensi —
empat hal yang menentukan pilihan untuk guardrail yang jalan di SETIAP pesan.

Bobot IndoBERT: indobenchmark/indobert-base-p1 (MIT, 124,5 juta parameter).
Dipakai sebagai PEMBANDING; model yang dirilis tetap `ner_service/model`.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ner_service"))
from evaluate import score_spans          # noqa: E402  — metrik yang sama persis
from indobert_backend import LABELS, L2I, MAXLEN, decode   # noqa: E402  — decoding yang sama
from ner_data import load_jsonl           # noqa: E402

MODEL_NAME = "indobenchmark/indobert-base-p1"
DATA = ROOT / "ner_service" / "data"
SETS = ["test_v8", "test_v7", "test_v6", "test_v5", "test_v4", "test_v3", "test_v2"]


def encode(tok, rows):
    """Teks + span karakter -> token id + label BIO, disejajarkan lewat offset mapping."""
    out = []
    for r in rows:
        enc = tok(r["text"], truncation=True, max_length=MAXLEN, return_offsets_mapping=True)
        labels = []
        for idx, (a, b) in enumerate(enc["offset_mapping"]):
            if a == b:                                    # token khusus ([CLS], [SEP])
                labels.append(-100)
                continue
            tag = "O"
            for s, e, lab in r["entities"]:
                if a >= s and b <= e:                     # subtoken di dalam span
                    tag = f"{'B' if a == s else 'I'}-{lab}"
                    break
            labels.append(L2I[tag])
        out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                    "labels": labels})
    return out


class Rows(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda seq, v: seq + [v] * (n - len(seq))       # noqa: E731
    return {
        "input_ids": torch.tensor([pad(b["input_ids"], 0) for b in batch]),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
        "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
    }


def spans_from(text, tok, model):
    """Prediksi -> {(start, end, label)}. Decoding-nya dipakai bersama service
    (ner_service/indobert_backend.py), supaya angka di sini = perilaku saat dilayani."""
    enc = tok(text, truncation=True, max_length=MAXLEN, return_offsets_mapping=True,
              return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        pred = model(**enc).logits[0].argmax(-1).tolist()
    return set(decode(text, pred, offsets))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / ".cache" / "indobert"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(torch.get_num_threads())

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        id2label={i: l for l, i in L2I.items()}, label2id=L2I)

    train_rows = load_jsonl(DATA / "train.jsonl")
    val_rows = load_jsonl(DATA / "val.jsonl")
    loader = DataLoader(Rows(encode(tok, train_rows)), batch_size=args.batch,
                        shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for i, batch in enumerate(loader, 1):
            loss = model(**batch).loss
            loss.backward(); opt.step(); opt.zero_grad()
            total += loss.item(); n += 1
            if i % 25 == 0:
                print(f"  epoch {ep} batch {i}/{len(loader)} loss={total / n:.4f}", flush=True)
        model.eval()
        v = score_spans(val_rows, lambda t: spans_from(t, tok, model))[0]["TOTAL"]
        print(f"epoch {ep}: loss={total / n:.4f}  val F2={v['f2']:.3f} R={v['r']:.3f} "
              f"bersih={v['neg'][0]}/{v['neg'][1]}", flush=True)
    latih_detik = round(time.perf_counter() - t0, 1)

    model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
    ukuran_mb = sum(f.stat().st_size for f in (out / "model").rglob("*")) / 2**20

    model.eval()
    hasil = {"model": MODEL_NAME, "epochs": args.epochs, "train_seconds": latih_detik,
             "size_mb": round(ukuran_mb, 1), "sets": {}}
    for name in SETS + ["val"]:
        rows = load_jsonl(DATA / (f"{name}.jsonl" if name != "val" else "val.jsonl"))
        t = score_spans(rows, lambda x: spans_from(x, tok, model))[0]["TOTAL"]
        hasil["sets"][name] = {k: (round(t[k], 4) if isinstance(t[k], float) else t[k])
                               for k in ("p", "r", "f1", "f2", "loose", "neg")}
        print(f"{name:8} P={t['p']:.2f} R={t['r']:.2f} F2={t['f2']:.2f} "
              f"longgar={t['loose']:.2f} bersih={t['neg'][0]}/{t['neg'][1]}", flush=True)

    contoh = "Halo saya Budi Santoso, rumah di Jalan Sudirman Jakarta, internet mati"
    for _ in range(5):
        spans_from(contoh, tok, model)
    waktu = []
    for _ in range(50):
        a = time.perf_counter(); spans_from(contoh, tok, model); waktu.append((time.perf_counter() - a) * 1000)
    hasil["latency_ms_p50"] = round(statistics.median(waktu), 2)
    hasil["latency_ms_p95"] = round(float(np.percentile(waktu, 95)), 2)
    print(f"\nlatency p50={hasil['latency_ms_p50']} ms  p95={hasil['latency_ms_p95']} ms  "
          f"ukuran={hasil['size_mb']} MB  latih={latih_detik}s")
    (out / "hasil.json").write_text(json.dumps(hasil, indent=2), encoding="utf-8")
    print("hasil ->", out / "hasil.json")


if __name__ == "__main__":
    main()
