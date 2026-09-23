"""
Siapkan word vectors Bahasa Indonesia untuk train.py (sekali saja).

    python ner_service/prepare_vectors.py                  # varian default: lower, 20 ribu
    python ner_service/prepare_vectors.py --attr ORTH --prune 50000 --out .cache/vec_orth50k

fastText cc.id.300 (Common Crawl + Wikipedia, lisensi CC BY-SA 3.0) berisi 2 juta kata x
300 angka (~1,2 GB terkompresi). Terlalu besar untuk image service kecil, jadi dipangkas:

  --truncate 200000 : hanya 200 ribu kata paling sering yang dibaca
  --prune N         : hanya N vektor yang disimpan; kata sisanya dipetakan ke vektor
                      terdekat di antara N itu

--attr LOWER (default sejak v7): kunci vektor huruf kecil. fastText menyimpan "Fatmawati",
"Agustinus", "Margonda" hanya dalam bentuk kapital, padahal chat pelanggan sering huruf
kecil semua — dengan ORTH, "fatmawati" tidak punya vektor. Kunci diturunkan ke huruf
kecil; bila ada beberapa bentuk ("Jakarta", "jakarta"), yang paling sering yang dipakai.
"""
import argparse
import gzip
import subprocess
import sys
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz"
CACHE = Path(__file__).resolve().parent.parent / ".cache"
RAW = CACHE / "cc.id.300.vec.gz"
TRUNCATE = 200_000


def lowercased(src: Path, dst: Path) -> None:
    """Tulis ulang TRUNCATE baris pertama dengan kunci huruf kecil (bentuk tersering menang)."""
    seen, lines = set(), []
    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as f:
        next(f)                                   # header "jumlah dimensi"
        for i, line in enumerate(f):
            if i >= TRUNCATE:
                break
            word, rest = line.split(" ", 1)
            key = word.lower()
            if key not in seen:                   # file urut frekuensi -> pertama = tersering
                seen.add(key)
                lines.append(f"{key} {rest}")
    with gzip.open(dst, "wt", encoding="utf-8") as f:
        f.write(f"{len(lines)} 300\n")
        f.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", choices=["ORTH", "LOWER"], default="ORTH")
    ap.add_argument("--prune", type=int, default=20_000)
    ap.add_argument("--out", default=str(CACHE / "vectors_id"))
    args = ap.parse_args()

    CACHE.mkdir(exist_ok=True)
    if not RAW.exists():
        print(f"mengunduh {URL} (~1,2 GB) ...")
        urllib.request.urlretrieve(URL, RAW)
    src = RAW
    if args.attr == "LOWER":
        src = CACHE / "cc.id.300.lower200k.vec.gz"
        if not src.exists():
            lowercased(RAW, src)
    subprocess.run([sys.executable, "-m", "spacy", "init", "vectors", "id", str(src), args.out,
                    "--truncate", str(TRUNCATE), "--prune", str(args.prune),
                    "--attr", args.attr, "--name", f"id_cc_fasttext_{args.attr.lower()}_{args.prune}"],
                   check=True)
    print(f"selesai -> {args.out}")


if __name__ == "__main__":
    main()
