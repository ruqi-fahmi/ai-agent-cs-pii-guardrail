"""
Pasca-proses tebakan model: pangkas kata umum di tepi span PERSON.

Model menebak entity per token; kadang ia menempelkan kata tanya/kata umum ke nama
("untuk pelanggan agustinus") atau menandai frasa tanpa nama sama sekali ("apakah
status"). Aturannya sengaja sederhana supaya mudah dijelaskan:

  1. Buang token di AWAL dan AKHIR span PERSON selama token itu kata umum
     (stopword Bahasa Indonesia bawaan spaCy + sedikit kosakata chat/CS).
  2. Kalau tidak ada yang tersisa, span dibuang.

Token di TENGAH tidak disentuh ("Nur binti Ahmad" tetap utuh). ADDRESS tidak disentuh,
karena alamat wajar mengandung kata umum ("tebet barat dalam jakarta").

Dicek otomatis oleh tests/test_ner_service.py: tidak ada satu pun nama di daftar
generator maupun val/test yang termasuk kata umum di bawah.
"""
from spacy.lang.id.stop_words import STOP_WORDS
from spacy.tokens import Doc, Span

# Kosakata chat & CS yang tidak ada di daftar stopword spaCy (bahasa baku).
CHAT_WORDS = {
    "gimana", "gmn", "gak", "ga", "nggak", "enggak", "tolong", "mohon", "kak", "kakak",
    "min", "admin", "gan", "bu", "mas", "mbak", "bang", "om", "tante", "ya", "yaa", "dong",
    "sih", "nih", "deh", "halo", "hai", "cek", "status", "nama", "pelanggan", "tagihan",
    "paket", "langganan", "modem", "wifi", "internet", "teknisi",
}
# Kata peran/relasi. Ditemukan saat mencoba model secara manual: setelah pemicu "atas nama",
# kata apa pun cenderung ditebak PERSON — termasuk "atas nama perusahaan" dan "atas nama
# suami". Yang PII di kalimat itu adalah NAMANYA, bukan kata perannya; kalau namanya ikut
# disebut ("atas nama suami saya, Budi"), nama itu tetap tersensor karena bukan kata umum.
# "anak" sengaja TIDAK dimasukkan: "Anak Agung" adalah nama Bali yang sungguhan.
ROLE_WORDS = {
    "perusahaan", "suami", "istri", "kantor", "yayasan", "pemilik", "almarhum", "almarhumah",
    "adik", "kakak", "saudara", "saudari", "atasan", "teman", "penyewa", "pengurus", "koperasi",
    "sekolah", "kampus", "keluarga", "orangtua", "ortu", "mertua", "ponakan", "sepupu",
    "tetangga", "penghuni", "kontrakan", "instansi", "lembaga", "toko", "warung",
}
# Singkatan gaya chat. Daftar stopword bawaan spaCy berisi bahasa baku ("kamu", "yang",
# "dengan"), sehingga bentuk singkatnya dianggap kata asing — dan kata asing di tengah
# kalimat adalah bentuk yang bagi model berarti nama. Ditemukan saat demo:
# "apa yg bisa dibantu?" -> "yg" ditandai PERSON.
ABBREVIATIONS = {
    "yg", "yng", "kmu", "km", "sy", "dgn", "utk", "tdk", "blm", "sdh", "udh", "krn", "jg",
    "bs", "dr", "dlm", "tp", "tpi", "spt", "hrs", "dpt", "msh", "sm", "kl", "klo", "trs",
    "lg", "pd", "dll", "dst", "yth", "gk", "gpp", "sblm", "stlh", "skrg", "bsk", "jgn",
}
COMMON = {w.lower() for w in STOP_WORDS} | CHAT_WORDS | ROLE_WORDS | ABBREVIATIONS


def _common(token) -> bool:
    return token.lower_ in COMMON or token.is_punct


def clean_ents(doc: Doc) -> list[Span]:
    out = []
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            out.append(ent)
            continue
        start, end = ent.start, ent.end
        while start < end and _common(doc[start]):
            start += 1
        while end > start and _common(doc[end - 1]):
            end -= 1
        if start < end:
            out.append(Span(doc, start, end, label=ent.label_))
    return out


def clean_spans(text: str, spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Versi clean_ents untuk span karakter mentah (backend non-spaCy).

    Aturannya sama: kata umum dipangkas dari tepi span PERSON, span yang habis dibuang.
    Pemenggalan katanya sederhana (spasi & tanda baca) karena backend lain tidak
    memberi token spaCy.
    """
    keluar = []
    for start, end, label in spans:
        if label != "PERSON":
            keluar.append((start, end, label))
            continue
        kata, i = [], start
        for bagian in text[start:end].split(" "):
            if bagian:
                kata.append((i, i + len(bagian), bagian))
            i += len(bagian) + 1
        a, b = 0, len(kata)
        umum = lambda k: k.strip(".,!?;:()").lower() in COMMON or not k.strip(".,!?;:()")  # noqa: E731
        while a < b and umum(kata[a][2]):
            a += 1
        while b > a and umum(kata[b - 1][2]):
            b -= 1
        if a < b:
            keluar.append((kata[a][0], kata[b - 1][1], label))
    return keluar
