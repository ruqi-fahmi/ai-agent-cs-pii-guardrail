"""
Evaluasi guardrail END-TO-END: seberapa banyak PII yang bocor lewat pipeline lengkap
(regex -> NER -> redaksi), bukan sekadar akurasi model NER.

    python scripts/eval_guardrail.py                          # NER dimuat in-process
    python scripts/eval_guardrail.py --url http://127.0.0.1:8020   # lewat NER Service

Per item PII di scripts/guardrail_eval.jsonl, hasilnya salah satu dari:
  - TERTUTUP  : tidak ada bagian yang tersisa di teks keluaran
  - SEBAGIAN  : sebagian tersisa (mis. nama marga tertinggal, atau 4 digit sisa NIK)
  - BOCOR     : nilainya masih utuh di teks keluaran
Plus "kata ikut tersensor": kata non-PII dari teks asli yang hilang dari keluaran
(ukuran over-redaction / FP).
"""
import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from cs_agent.guardrails import callback, ner_client   # noqa: E402

DATA = Path(__file__).with_name("guardrail_eval.jsonl")
REPORT = ROOT / "docs" / "guardrail-evaluation.md"
# Kata >= 2 huruf: dulu >= 3, sehingga "Rp" yang ikut tersensor tidak terhitung
# (titik buta yang ditemukan test browser, 22 Sep 2026).
TOKEN = re.compile(r"[A-Za-z]{2,}|\d{3,}")


def tokens(s: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(s)}


def use_local_model() -> None:
    import spacy
    sys.path.insert(0, str(ROOT / "ner_service"))
    from postprocess import clean_ents     # sama persis dengan yang dikirim NER Service
    nlp = spacy.load(ROOT / "ner_service" / "model")

    async def detect(text):
        return [{"label": e.label_, "text": e.text, "start": e.start_char, "end": e.end_char}
                for e in clean_ents(nlp(text))]
    ner_client.detect = detect


async def run(rows):
    per_type = defaultdict(lambda: defaultdict(int))
    details, over = [], []
    for r in rows:
        text = r["text"]
        clean, _ = await callback.redact_text(text)
        clean_tokens = tokens(clean)
        for value, ptype in r["pii"]:
            vt = tokens(value)
            left = vt & clean_tokens
            if value in clean:
                status = "BOCOR"
            elif left:
                status = "SEBAGIAN"
            else:
                status = "TERTUTUP"
            per_type[ptype][status] += 1
            if status != "TERTUTUP":
                details.append((status, ptype, value, sorted(left), clean))
        # over-redaction: kata non-PII yang hilang dari keluaran
        pii_tokens = set().union(*(tokens(v) for v, _ in r["pii"])) if r["pii"] else set()
        for t in tokens(text) - pii_tokens - clean_tokens:
            over.append((t, text))
    return per_type, details, over


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="URL NER Service; tanpa ini model dimuat in-process")
    args = ap.parse_args()
    if args.url:
        ner_client._client = ner_client.httpx.AsyncClient(base_url=args.url, timeout=5)
        mode = f"NER Service {args.url}"
    else:
        use_local_model()
        mode = "model NER in-process"

    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_type, details, over = asyncio.run(run(rows))

    order = ["NIK", "EMAIL", "PHONE", "PERSON", "ADDRESS"]
    lines = ["| Jenis | Lapis | Jumlah | Tertutup | Sebagian | Bocor utuh |", "|---|---|---|---|---|---|"]
    tot = defaultdict(int)
    for t in order:
        c = per_type[t]
        n = sum(c.values())
        for k in ("TERTUTUP", "SEBAGIAN", "BOCOR"):
            tot[k] += c[k]
        layer = "regex" if t in ("NIK", "EMAIL", "PHONE") else "NER"
        lines.append(f"| {t} | {layer} | {n} | {c['TERTUTUP']} | {c['SEBAGIAN']} | {c['BOCOR']} |")
    n_all = sum(tot.values())
    lines.append(f"| **Total** | | **{n_all}** | **{tot['TERTUTUP']}** ({tot['TERTUTUP']/n_all:.0%}) | "
                 f"**{tot['SEBAGIAN']}** | **{tot['BOCOR']}** ({tot['BOCOR']/n_all:.0%}) |")
    table = "\n".join(lines)
    print(f"Mode: {mode}\n{len(rows)} kalimat, {n_all} item PII\n\n{table}")
    for status, ptype, value, left, clean in details:
        print(f"  {status:8} {ptype:8} {value!r} sisa={left}")
    print(f"\nKata non-PII ikut tersensor: {len(over)} -> {[t for t, _ in over]}")

    det = ["| Status | Jenis | PII asli | Sisa di keluaran | Teks keluaran |", "|---|---|---|---|---|"]
    det += [f"| {s} | {p} | `{v}` | {', '.join(l) or '—'} | {c} |" for s, p, v, l, c in details]
    ov = ["| Kata | Kalimat asli |", "|---|---|"] + [f"| `{t}` | {x} |" for t, x in over]
    REPORT.write_text(
        "# Evaluasi Guardrail End-to-End\n\n"
        "> Ditulis ulang otomatis oleh `python scripts/eval_guardrail.py`.\n\n"
        f"Pipeline lengkap (regex → NER → redaksi) dijalankan pada "
        f"[`scripts/guardrail_eval.jsonl`](../scripts/guardrail_eval.jsonl): {len(rows)} kalimat "
        f"chat CS campuran, {n_all} item PII, termasuk kalimat tanpa PII yang mirip PII "
        "(nominal tagihan, order ID 20 digit, nama kota). Nama & alamat di set ini tidak ada "
        f"di data latih maupun set uji NER. Mode: {mode}.\n\n"
        "Pertanyaan yang dijawab: **berapa PII yang benar-benar sampai ke LLM?** — bukan "
        "sekadar akurasi model.\n\n"
        "## Hasil\n\n" + table + "\n\n"
        "- **Tertutup**: tidak ada bagian nilai asli yang tersisa.\n"
        "- **Sebagian**: sebagian token tersisa (mis. marga tertinggal) — bocor parsial.\n"
        "- **Bocor utuh**: nilai asli masih lengkap di teks yang dikirim ke LLM.\n\n"
        "## Item yang tidak tertutup penuh\n\n" + ("\n".join(det) if details else "_Tidak ada._") + "\n\n"
        f"## Over-redaction — kata non-PII yang ikut tersensor ({len(over)})\n\n"
        + ("\n".join(ov) if over else "_Tidak ada._") + "\n",
        encoding="utf-8")
    print(f"\nLaporan: {REPORT}")


if __name__ == "__main__":
    main()
