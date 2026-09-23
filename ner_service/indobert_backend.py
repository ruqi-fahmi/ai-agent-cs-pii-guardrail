"""
Backend alternatif: fine-tune IndoBERT (transformers) sebagai pengganti model spaCy.

Dipakai oleh NER Service bila `NER_BACKEND=indobert`, dan oleh
`benchmark/indobert_compare.py` saat melatih & mengukurnya.

**Model yang dirilis tetap yang spaCy** (dilatih dari nol, 40 MB). Backend ini ada supaya
klaim di PRD §6 bisa diuji dengan angka, bukan diperdebatkan: IndoBERT lebih akurat, tapi
12x lebih besar, 2,8x lebih boros RAM, dan ~5x lebih lambat. Lihat docs/ner-iterasi.md.

Bobot: indobenchmark/indobert-base-p1 (MIT). `transformers` dan `torch` TIDAK ada di
requirements.txt service — backend ini hanya bisa dipakai bila keduanya dipasang manual
(lihat requirements-indobert.txt).
"""
from pathlib import Path

LABELS = ["O", "B-PERSON", "I-PERSON", "B-ADDRESS", "I-ADDRESS"]
L2I = {l: i for i, l in enumerate(LABELS)}
MAXLEN = 160


class IndoBertNer:
    """Kontrak yang sama dengan pipeline spaCy: .entities(teks) -> [(start, end, label)]."""

    def __init__(self, model_dir: str | Path):
        import torch                                     # impor di sini: opsional
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.version = f"indobert_ner_id-{getattr(self.model.config, 'model_version', '1.0.0')}"

    def entities(self, text: str) -> list[tuple[int, int, str]]:
        enc = self.tok(text, truncation=True, max_length=MAXLEN,
                       return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        with self._torch.no_grad():
            pred = self.model(**enc).logits[0].argmax(-1).tolist()
        return decode(text, pred, offsets)


def decode(text: str, pred: list[int], offsets: list[list[int]]) -> list[tuple[int, int, str]]:
    """Label per subtoken -> span karakter. B- memulai span baru, I- menyambung."""
    spans: list[tuple[int, int, str]] = []
    cur = None
    for tag_id, (a, b) in zip(pred, offsets):
        if a == b:                                       # [CLS] / [SEP] / padding
            continue
        tag = LABELS[tag_id] if 0 <= tag_id < len(LABELS) else "O"
        if tag == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        pos, lab = tag.split("-")
        if pos == "B" or cur is None or cur[2] != lab:
            if cur:
                spans.append(cur)
            cur = (a, b, lab)
        else:
            cur = (cur[0], b, lab)
    if cur:
        spans.append(cur)
    # Subtoken bisa membawa spasi di tepi; rapikan supaya offset sebanding dengan spaCy.
    rapi = []
    for s, e, lab in spans:
        potongan = text[s:e]
        s += len(potongan) - len(potongan.lstrip())
        e -= len(potongan) - len(potongan.rstrip())
        if s < e:
            rapi.append((s, e, lab))
    return rapi
