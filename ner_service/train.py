"""
Latih model NER PERSON/ADDRESS. Bobot NER dilatih dari nol; sejak v6 ditambah
word vectors fastText Bahasa Indonesia sebagai FITUR (lihat docs/ner-iterasi.md).

    python ner_service/data/generate_train.py
    python ner_service/prepare_vectors.py        # sekali: unduh & pangkas fastText
    python ner_service/train.py                  # --no-vectors = tanpa vektor (pembanding)
    python ner_service/sweep.py                  # banyak konfigurasi x seed, pilih di val

Alur:
  1. pipeline berisi tokenizer Bahasa Indonesia + tabel word vectors (kamus kata ->
     300 angka yang mewakili makna kata, dipelajari fastText dari Common Crawl)
  2. tambah komponen "ner" -> jaringan saraf kecil yang bobotnya masih acak. Ia membaca
     vektor kata sebagai petunjuk tambahan: "apakah" & "status" dekat dengan kata umum,
     "fahmi" dekat dengan nama orang — padahal ketiganya tidak pernah ada di data latih
  3. ulangi N epoch: tunjukkan contoh train, model menebak, hitung loss
     (seberapa salah tebakannya), perbaiki bobot sedikit
  4. tiap epoch ukur skor di data VALIDATION (ditulis tangan); simpan epoch terbaik
     -> mencegah memakai model yang sudah mulai overfit (hafal template train)

Tiga jenis data dengan peran berbeda:
  - dev.jsonl   : pecahan generator sintetis — hanya memantau apakah model belajar.
                  Skornya selalu ~1.00 karena polanya sama dengan train; TIDAK dipakai
                  untuk memilih model.
  - val.jsonl   : ditulis tangan, dipakai MEMILIH epoch terbaik.
  - test_v4.jsonl: ditulis tangan, TIDAK disentuh di sini — hanya evaluate.py, sekali.

Metrik pemilihan: F2 = F-score yang menimbang recall 2x lebih penting daripada
precision. Untuk guardrail, nama yang lolos (FN) = bocor, lebih mahal daripada kata
biasa yang ikut disensor (FP).
"""
import argparse
import json
import random
import time
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import compounding, fix_random_seed, minibatch

from evaluate import score
from ner_data import load_jsonl

BASE = Path(__file__).parent
DATA = BASE / "data"
MODEL_DIR = BASE / "model"
VECTORS_DIR = BASE.parent / ".cache" / "vectors_id"    # hasil prepare_vectors.py --attr ORTH
LABELS = ["PERSON", "ADDRESS"]
N_EPOCHS = 20
DROPOUT = 0.3   # matikan acak 30% neuron saat latihan -> memaksa model tidak bergantung
                # pada satu petunjuk saja (salah satu cara melawan overfitting)
SEED = 4   # v8: terpilih dari sweep 8 seed berdasarkan val (lihat docs/ner-iterasi.md)
BETA = 2.0      # F-beta: beta > 1 = recall lebih dipentingkan
MODEL_VERSION = "3.2.0"


def to_examples(nlp, rows):
    examples = []
    for r in rows:
        doc = nlp.make_doc(r["text"])
        for start, end, label in r["entities"]:
            # NER spaCy bekerja per TOKEN. Entity yang batasnya jatuh di tengah
            # token tidak bisa dipelajari -> gagal keras daripada diam-diam salah.
            if doc.char_span(start, end, label=label) is None:
                raise ValueError(f"Batas entity tidak pas token: {r['text'][start:end]!r} "
                                 f"di {r['text']!r}")
        examples.append(Example.from_dict(doc, {"entities": r["entities"]}))
    return examples


def ner_config(use_vectors: bool, width: int, embed_size: int) -> dict:
    # Arsitektur bawaan spaCy; yang bisa diubah: pretrained_vectors, width, embed_size.
    return {"model": {
        "@architectures": "spacy.TransitionBasedParser.v2", "state_type": "ner",
        "extra_state_tokens": False, "hidden_width": 64, "maxout_pieces": 2,
        "use_upper": True, "nO": None,
        "tok2vec": {"@architectures": "spacy.HashEmbedCNN.v2",
                    "pretrained_vectors": use_vectors, "width": width, "depth": 4,
                    "embed_size": embed_size, "window_size": 1, "maxout_pieces": 3,
                    "subword_features": True}}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vectors", action="store_true", help="tanpa word vectors (pembanding)")
    ap.add_argument("--vectors", default=str(VECTORS_DIR), help="folder hasil prepare_vectors.py")
    ap.add_argument("--data", default=str(DATA), help="folder train.jsonl & dev.jsonl")
    ap.add_argument("--out", default=str(MODEL_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--width", type=int, default=128)       # v7: sweep E (96 -> 128)
    ap.add_argument("--embed-size", type=int, default=5000)  # v7: sweep E (2000 -> 5000)
    args = ap.parse_args()
    out_dir, data_dir = Path(args.out), Path(args.data)
    use_vectors = not args.no_vectors

    fix_random_seed(args.seed)
    random.seed(args.seed)

    if use_vectors:
        if not Path(args.vectors).exists():
            raise SystemExit(f"{args.vectors} belum ada — jalankan prepare_vectors.py dulu")
        nlp = spacy.load(args.vectors)     # tokenizer "id" + tabel vektor, tanpa komponen
    else:
        nlp = spacy.blank("id")
    ner = nlp.add_pipe("ner", config=ner_config(use_vectors, args.width, args.embed_size))
    for label in LABELS:
        ner.add_label(label)

    train = to_examples(nlp, load_jsonl(data_dir / "train.jsonl"))
    dev = to_examples(nlp, load_jsonl(data_dir / "dev.jsonl"))
    val_rows = load_jsonl(DATA / "val.jsonl")   # val selalu dari data/ — ditulis tangan
    print(f"train={len(train)} dev={len(dev)} val={len(val_rows)}")

    optimizer = nlp.initialize(lambda: train)
    best, log = None, []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        losses = {}
        for batch in minibatch(train, size=compounding(4.0, 32.0, 1.001)):
            nlp.update(batch, drop=args.dropout, sgd=optimizer, losses=losses)
        dev_f = float(nlp.evaluate(dev)["ents_f"] or 0.0)
        # Val dinilai PERSIS seperti yang dikirim service (model + postprocess.clean_ents).
        t = score(nlp, val_rows)[0]["TOTAL"]
        clean, n_neg = t["neg"]
        entry = {"epoch": epoch, "loss": round(float(losses["ner"]), 2),
                 "dev_f1": round(dev_f, 4), "val_p": round(t["p"], 4),
                 "val_r": round(t["r"], 4), "val_f2": round(t["f2"], 4),
                 "val_clean": clean, "val_neg": n_neg}
        log.append(entry)
        key = (entry["val_f2"], entry["val_r"], clean)   # seri F2 -> recall, lalu kalimat bersih
        mark = ""
        if best is None or key > best:
            best = key
            nlp.meta.update({"name": "pii_ner_id", "version": MODEL_VERSION,
                             "description": "NER PERSON/ADDRESS, dilatih dari nol"
                             + (" + fitur fastText cc.id.300" if use_vectors else "")})
            nlp.to_disk(out_dir)
            mark = "  <- terbaik, disimpan"
        print(f"epoch {epoch:2d}  loss={losses['ner']:8.2f}  dev_f1={dev_f:.3f}  "
              f"val P={t['p']:.2f} R={t['r']:.2f} F2={t['f2']:.3f} bersih={clean}/{n_neg}{mark}",
              flush=True)

    best_epoch = max(log, key=lambda e: (e["val_f2"], e["val_r"], e["val_clean"]))
    (out_dir / "training_log.json").write_text(
        json.dumps({"epochs": log, "best_epoch": best_epoch,
                    "selection": f"F{BETA:g} on val.jsonl (model + postprocess)",
                    "train_seconds": round(time.perf_counter() - t0, 1),
                    "n_train": len(train), "n_dev": len(dev), "n_val": len(val_rows),
                    "config": {k: v for k, v in vars(args).items() if k != "out"}},
                   indent=2),
        encoding="utf-8")
    print(f"selesai. epoch terbaik={best_epoch['epoch']} val F2={best[0]:.3f} -> {out_dir}")


if __name__ == "__main__":
    main()
