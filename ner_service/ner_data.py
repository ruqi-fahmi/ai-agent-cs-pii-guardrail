"""Loader dataset NER — dipakai train.py & evaluate.py."""
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Baca dataset -> list {"text", "entities": [(start, end, label), ...]}.

    Dua format entity diterima:
      - [start, end, label]  (keluaran generate_train.py)
      - [teks_entity, label] (test.jsonl, ditulis tangan — offset dihitung di sini
        supaya tidak ada salah hitung karakter saat menulis manual)
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text, ents = row["text"], []
            for ent in row["entities"]:
                if len(ent) == 3:
                    ents.append((ent[0], ent[1], ent[2]))
                else:
                    value, label = ent
                    start = text.find(value)
                    if start < 0:
                        raise ValueError(f"{path.name}:{line_no}: '{value}' tidak ada di teks")
                    ents.append((start, start + len(value), label))
            rows.append({"text": text, "entities": ents})
    return rows
