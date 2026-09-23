"""
Ukur akurasi model NER pada data yang ditulis tangan.

    python ner_service/evaluate.py                    # model aktif, tulis laporan
    python ner_service/evaluate.py --model <dir> --no-report   # bandingkan model lain

Data (lihat docs/ner-iterasi.md untuk sejarahnya):
  - test_v7.jsonl : BUTA — ditulis sebelum v9 dibuat, dievaluasi sekali. Ini angka utama.
                    Fokus: alamat tanpa awalan "Jl." (kelurahan/kecamatan + kota).
  - test_v6.jsonl : buta untuk v8 — nama hari & bulan, kini pembanding.
  - test_v5.jsonl : buta untuk v7 — nama sesudah "pelanggan/buat/punya", alamat tanpa
                    awalan + arah mata angin; kini pembanding.
  - test_v4.jsonl : buta untuk v6 — kalimat tanya CS & alamat tanpa "Jl.", kini pembanding.
  - test_v3.jsonl : buta untuk v5 — fokus kalimat curhat tanpa PII, kini pembanding.
  - test_v2.jsonl : buta untuk v4, kini pembanding.
  - test.jsonl    : test v1 — sudah dilihat berkali-kali selama iterasi v1-v3, jadi
                    "terkontaminasi"; ditampilkan hanya sebagai pembanding.
  - val.jsonl     : dipakai memilih epoch di train.py; skornya optimistis.

Metrik per label, dihitung manual supaya jelas asal-usulnya:
  TP = entity yang ditebak model DAN benar (label & batas persis sama)
  FP = model bilang entity, ternyata bukan / batasnya salah   -> salah sensor
  FN = entity asli yang tidak ditebak dengan tepat           -> BOCOR
  precision = TP / (TP + FP),  recall = TP / (TP + FN)
  F1 = rata-rata harmonik precision & recall; F2 = versi yang menimbang recall 2x

Plus "recall longgar": entity asli yang setidaknya TERSENTUH tebakan berlabel sama —
untuk guardrail, sebagian nama yang tertutup lebih baik daripada tidak sama sekali.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import spacy

from ner_data import load_jsonl
from postprocess import clean_ents

BASE = Path(__file__).parent
REPORT = BASE.parent / "docs" / "ner-evaluation.md"
SETS = [
    ("test_v8", "test_v8.jsonl", "**buta** untuk model ini — batas alamat & frasa lokasi umum"),
    ("test_v7", "test_v7.jsonl", "buta v9 — alamat tanpa awalan, angka utama saat model dipilih"),
    ("test_v6", "test_v6.jsonl", "buta v8 — nama hari & bulan, pembanding"),
    ("test_v5", "test_v5.jsonl", "buta v7 — posisi nama & alamat tanpa awalan, pembanding"),
    ("test_v4", "test_v4.jsonl", "buta v6 — kalimat tanya & alamat tanpa awalan, pembanding"),
    ("test_v3", "test_v3.jsonl", "buta v5 — kalimat curhat, pembanding"),
    ("test_v2", "test_v2.jsonl", "buta v4 — sudah dilihat sekali, pembanding"),
    ("test_v1", "test.jsonl", "terkontaminasi — pembanding"),
    ("val", "val.jsonl", "dipakai memilih epoch — optimistis"),
]


def score(nlp, rows, raw: bool = False):
    """Skor model spaCy. Pembungkus tipis di atas score_spans."""
    def predict(text: str) -> set:
        doc = nlp(text)
        ents = doc.ents if raw else clean_ents(doc)   # default = persis seperti yang dikirim service
        return {(e.start_char, e.end_char, e.label_) for e in ents}
    return score_spans(rows, predict)


def score_spans(rows, predict):
    """Metrik yang SAMA untuk model apa pun: `predict(teks) -> {(start, end, label)}`.

    Dipisah supaya model pembanding (mis. fine-tune IndoBERT di benchmark/) dinilai
    dengan kode yang persis sama, bukan implementasi metrik yang berbeda.
    """
    tp, fp, fn, touched, gold_count = (defaultdict(int) for _ in range(5))
    errors = []
    neg_total = neg_clean = 0
    for r in rows:
        text = r["text"]
        gold = {(s, e, l) for s, e, l in r["entities"]}
        pred = predict(text)
        if not gold:
            neg_total += 1
            neg_clean += not pred
        for s, e, l in gold:
            gold_count[l] += 1
            if any(pl == l and ps < e and s < pe for ps, pe, pl in pred):
                touched[l] += 1
        for item in pred & gold:
            tp[item[2]] += 1
        for s, e, l in pred - gold:
            fp[l] += 1
            errors.append(("FP", l, text[s:e], text))
        for s, e, l in gold - pred:
            fn[l] += 1
            errors.append(("FN", l, text[s:e], text))

    rows_out = {}
    for label in ["PERSON", "ADDRESS", "TOTAL"]:
        if label == "TOTAL":
            t, p, n = sum(tp.values()), sum(fp.values()), sum(fn.values())
            g, tc = sum(gold_count.values()), sum(touched.values())
        else:
            t, p, n, g, tc = tp[label], fp[label], fn[label], gold_count[label], touched[label]
        prec = t / (t + p) if t + p else 0.0
        rec = t / (t + n) if t + n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
        rows_out[label] = dict(gold=g, tp=t, fp=p, fn=n, p=prec, r=rec, f1=f1, f2=f2,
                               loose=tc / g if g else 0.0)
    # Kalimat tanpa PII yang tidak disensor sama sekali — FP yang paling terasa di demo.
    rows_out["TOTAL"]["neg"] = (neg_clean, neg_total)
    return rows_out, errors


def table(res) -> str:
    lines = ["| Label | Gold | TP | FP | FN | Precision | Recall | F1 | F2 | Recall longgar |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, m in res.items():
        lines.append(f"| {label} | {m['gold']} | {m['tp']} | {m['fp']} | {m['fn']} | "
                     f"{m['p']:.2f} | {m['r']:.2f} | {m['f1']:.2f} | {m['f2']:.2f} | {m['loose']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(BASE / "model"))
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--raw", action="store_true", help="model saja, tanpa postprocess.clean_ents")
    args = ap.parse_args()

    nlp = spacy.load(args.model)
    version = f"{nlp.meta.get('name')}-{nlp.meta.get('version')}"
    sections, primary_errors = [], []
    for key, fname, note in SETS:
        rows = load_jsonl(BASE / "data" / fname)
        res, errors = score(nlp, rows, raw=args.raw)
        t = res["TOTAL"]
        nc, nt = t["neg"]
        print(f"{key:8} n={len(rows):2d}  P={t['p']:.2f} R={t['r']:.2f} F1={t['f1']:.2f} "
              f"F2={t['f2']:.2f} longgar={t['loose']:.2f}  bersih={nc}/{nt}  kesalahan={len(errors)}")
        sections.append(f"### {key} — `{fname}` ({len(rows)} kalimat, {note})\n\n{table(res)}\n\n"
                        f"Kalimat tanpa PII yang tetap bersih: **{nc}/{nt}**\n")
        if key == "test_v7":
            primary_errors = errors
            for k, l, frag, _ in errors:
                print(f"    {k} {l:8} {frag!r}")

    if args.no_report or args.raw:
        return
    err_lines = ["| Jenis | Label | Potongan | Kalimat |", "|---|---|---|---|"]
    err_lines += [f"| {k} | {l} | `{frag}` | {t} |" for k, l, frag, t in primary_errors]
    REPORT.write_text(
        "# Evaluasi Model NER\n\n"
        f"> Model `{version}`. File ini ditulis ulang otomatis oleh "
        "`python ner_service/evaluate.py`.\n\n"
        "Semua data uji ditulis tangan, dengan nama & alamat yang **tidak ada** di data latih. "
        "Sejarah iterasi dan catatan metodologi: [ner-iterasi.md](ner-iterasi.md).\n\n"
        "## Skor\n\n" + "\n".join(sections) + "\n"
        "- **Exact match**: benar hanya bila label DAN batas awal-akhir persis sama.\n"
        "- **F2**: F-score yang menimbang recall 2× — metrik pemilihan model, karena untuk "
        "guardrail FN (bocor) lebih mahal daripada FP (salah sensor).\n"
        "- **Recall longgar**: entity asli yang setidaknya tersentuh tebakan berlabel sama.\n\n"
        "## Kesalahan di test_v7\n\n"
        "- **FN** = entity asli tidak tertebak dengan tepat → berpotensi bocor.\n"
        "- **FP** = tebakan yang salah (bukan entity, atau batasnya meleset) → salah sensor.\n"
        "- Satu entity yang batasnya meleset tercatat dua kali: FN (yang benar) + FP (yang ditebak).\n\n"
        + "\n".join(err_lines) + "\n",
        encoding="utf-8")
    print(f"\nLaporan: {REPORT}")


if __name__ == "__main__":
    main()
