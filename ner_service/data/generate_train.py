"""
Generator dataset latih NER (sintetis).

Kenapa sintetis: tidak ada data chat CS berlabel, dan data pelanggan asli tidak
boleh dipakai (PII!). Kalimat dirakit dari template x daftar nama x daftar alamat,
sehingga posisi entity (start, end) DIKETAHUI PASTI tanpa melabeli manual.

Kelemahan yang disadari: model bisa sekadar "menghafal template". Karena itu model
DIPILIH memakai val.jsonl dan DIUJI memakai test_v3.jsonl — keduanya ditulis tangan,
dengan nama/alamat yang sengaja TIDAK ada di daftar di bawah (lihat docs/ner-iterasi.md).

Output: train.jsonl & dev.jsonl, tiap baris {"text": ..., "entities": [[start, end, label], ...]}
"""
import argparse
import json
import random
from pathlib import Path

SEED = 42
N_SENTENCES = 4000
DEV_RATIO = 0.10
LOWERCASE_RATIO = 0.3   # chat asli sering huruf kecil semua
COMPOSED_RATIO = 0.35   # v5: porsi kalimat rakitan (curhat, rupiah) — lihat build_composed
OUT_DIR = Path(__file__).parent

# Keragaman nama: Jawa, Sunda, Batak, Bali, Minang, Tionghoa-Indonesia, Arab-Indonesia,
# nama baptis. Nama di val/test sengaja tidak dimasukkan.
FIRST_NAMES = [
    "Budi", "Siti", "Agus", "Dewi", "Rudi", "Ani", "Joko", "Rina", "Andi", "Sri",
    "Hendra", "Lestari", "Bambang", "Wulan", "Dedi", "Fitri", "Yusuf", "Nur",
    "Rizky", "Putri", "Fajar", "Intan", "Arif", "Maya", "Eko", "Ratna", "Doni",
    "Sari", "Taufik", "Indah", "Wahyu", "Dian", "Ahmad", "Nurul", "Bayu", "Ayu",
    "Hadi", "Mega", "Irfan", "Yuni", "Galih", "Novi", "Reza", "Tika", "Imam",
    "Kadek", "Putu", "Gede", "Luh", "Yosef", "Markus", "Paulus", "Agnes", "Veronika",
    "Lukas", "Hasan", "Fatimah", "Khadijah", "Zulkifli", "Ilham", "Vina", "Citra",
    "Robert", "Hendro", "Suwarni", "Poniman", "Sutrisno", "Jefri", "Wenny", "Chandra",
    "Sinta", "Yanti", "Rosa", "Tan",
]
LAST_NAMES = [
    "Santoso", "Wijaya", "Saputra", "Lestari", "Hidayat", "Pratama", "Kurniawan",
    "Nugroho", "Susanti", "Setiawan", "Siregar", "Nasution", "Harahap", "Simanjuntak",
    "Wibowo", "Rahmawati", "Purnomo", "Hakim", "Maharani", "Gunawan", "Permana",
    "Firmansyah", "Sembiring", "Lubis", "Utami", "Hasibuan", "Anggraini", "Ramadhan",
    "Sihombing", "Pakpahan", "Silalahi", "Napitupulu", "Hutapea", "Ginting", "Tarigan",
    "Sinaga", "Pasaribu", "Gultom", "Suyasa", "Artawan", "Mahendra", "Halim", "Tjandra",
    "Hidayatullah", "Alatas", "Kusuma", "Pranoto", "Sugiarto", "Tanjung", "Batubara",
    "Rangkuti", "Daulay",
]
# v5: v4 hanya punya ~74 nama depan, sehingga model cenderung MENGHAFAL nama. Setelah
# kalimat curhat ditambahkan, nama huruf kecil yang belum pernah dilihat ("fauzan
# alfarizi") mulai lolos. Daftar diperluas supaya model belajar bentuk nama, bukan
# daftar nama. Nama di val/test tetap dikeluarkan (dicek oleh main()).
FIRST_NAMES += [
    "Aditya", "Cahya", "Danang", "Erlangga", "Fadli", "Guntur", "Harun",
    "Iwan", "Jaka", "Krisna", "Lukman", "Miftah", "Naufal", "Oki", "Pandu", "Qori",
    "Rangga", "Satria", "Teguh", "Vicky", "Wisnu", "Yoga", "Zaki", "Anisa",
    "Bella", "Cindy", "Dinda", "Elsa", "Febri", "Gita", "Hana", "Ira", "Jihan",
    "Kiki", "Laras", "Mira", "Nadya", "Olivia", "Prita", "Rahma", "Salsa", "Tiara",
    "Ulfa", "Vera", "Winda", "Yulia", "Zahra", "Asep", "Ujang", "Cecep", "Dadang",
    "Euis", "Neneng", "Iis", "Entis", "Nyoman", "Ngurah", "Ayu",
    "Horas", "Parlindungan", "Togar", "Butet", "Tiurma", "Hotman", "Sahat", "Rudolf",
    "Engkus", "Afrizal", "Datuk", "Syafri", "Yusnidar", "Rosmaini", "Aulia",
    "Hendrik", "Kornelius", "Stefani", "Fransiska", "Benediktus",
    "Aloysius", "Maximilian", "Elisabet", "Ignatius", "Susana", "Mikael", "Albert",
    "Linda", "Ming", "Hui", "Liong", "Mei", "Acong", "Aming", "Siu", "Kwee",
    "Abdullah", "Husein", "Salim", "Faisal", "Syarifah", "Habib", "Zainab",
    "Sukarni", "Paijo", "Tukiyem", "Painem", "Sukijan", "Legiman", "Warsini",
    "Samuel", "Daniel", "Natalia", "Jessica", "Michael", "Stevanus",
    "Andika", "Rizal", "Fikri", "Hafiz", "Nabila", "Syifa", "Alya", "Keisha",
]
LAST_NAMES += [
    "Wicaksono", "Prasetyo", "Suryadi", "Hartono", "Budiman", "Sulistyo", "Handoko",
    "Rahardjo", "Susilo", "Kartika", "Puspita", "Safitri", "Febriani",
    "Suryani", "Hermawan", "Kusnadi", "Sutanto", "Sanjaya", "Hutagaol",
    "Panjaitan", "Marpaung", "Situmeang", "Damanik", "Purba", "Saragih", "Sitepu",
    "Pinem", "Barus", "Sitorus", "Aritonang", "Nainggolan", "Hutabarat", "Tampubolon",
    "Siagian", "Lumbantobing", "Wardana", "Adnyana", "Suardika",
    "Dharma", "Wiguna", "Pratiwi", "Syahrial", "Chaniago", "Piliang", "Sikumbang",
    "Koto", "Tanjaya", "Gunadi", "Salim", "Kosasih", "Lie", "Tan", "Oei", "Liem",
    "Assegaf", "Shihab", "Baswedan", "Bafadal", "Mandagi", "Rumengan",
    "Lumowa", "Pattiasina", "Latuconsina", "Rumbiak", "Mandacan", "Nggadas",
    "Hidayati", "Rosyidi", "Mubarok", "Fauzi", "Hamzah", "Ismail", "Yahya", "Zulkarnain",
]
BATAK_MARGA = ["Sihombing", "Pakpahan", "Silalahi", "Napitupulu", "Hutapea", "Ginting",
               "Tarigan", "Sinaga", "Pasaribu", "Gultom", "Siregar", "Nasution"]

STREET_PREFIX = ["Jalan", "Jl.", "Jln", "Jl", "Jalan Raya", "Jl. Raya"]
STREET_NAMES = [
    "Sudirman", "Thamrin", "Gatot Subroto", "Diponegoro", "Ahmad Yani", "Merdeka",
    "Pahlawan", "Veteran", "Gajah Mada", "Hayam Wuruk", "Imam Bonjol", "Kartini",
    "Pemuda", "Asia Afrika", "Braga", "Dago", "Malioboro", "Pandanaran", "Setiabudi",
    "Kenanga", "Mawar", "Cempaka", "Anggrek", "Dahlia", "Flamboyan", "Cendana",
    "Wolter Monginsidi", "Slamet Riyadi", "Pangeran Antasari", "Siliwangi", "Rasuna Said",
    "Kusumanegara", "Dr. Sutomo", "Sam Ratulangi", "Kapten Muslim", "Adisucipto",
    "Urip Sumoharjo", "Tuanku Tambusai", "Sultan Agung", "MH Thamrin",
]
CITIES = [
    "Jakarta", "Bandung", "Surabaya", "Medan", "Semarang", "Yogyakarta", "Makassar",
    "Palembang", "Bekasi", "Tangerang", "Depok", "Bogor", "Malang", "Denpasar",
    "Pekanbaru", "Padang", "Balikpapan", "Pontianak", "Manado", "Solo", "Cirebon",
    "Jambi", "Kupang", "Mataram", "Samarinda", "Tasikmalaya", "Purwokerto", "Binjai",
    "Kediri", "Magelang", "Jakarta Selatan", "Jakarta Barat",
]
KOMPLEKS = ["Perumahan Griya Asri", "Perum Taman Sari", "Komplek Bumi Indah",
            "Perumahan Citra Garden", "Komplek Villa Mutiara", "Perum Graha Permai",
            "Perumahan Puri Indah", "Komplek Setia Budi Regency"]
GANG = ["Gang Melati", "Gg. Kelapa", "Gang Sawo", "Gg Nangka", "Gang Durian", "Gang Buntu",
        "Gg. Rukun"]
KELURAHAN = ["Kebon Jeruk", "Cempaka Putih", "Sukajadi", "Tegalsari", "Petisah",
             "Lowokwaru", "Tembalang", "Menteng", "Cibeunying", "Rungkut"]
KAMPUNG = ["Kampung Bali", "Kampung Sawah", "Kampung Baru", "Kampung Rambutan",
           "Kp. Cilandak", "Kp. Babakan"]
DESA = ["Desa Mekarsari", "Desa Sukajaya", "Desa Tanjungsari", "Desa Wonosari"]
DUSUN = ["Dusun Sidomulyo", "Dusun Ngemplak", "Dusun Tegalrejo"]
KECAMATAN = ["Ciputat", "Sunggal", "Kuta", "Cimanggis", "Mlati", "Kartasura"]
# v6: alamat TANPA awalan ("cilandak raya jakarta selatan"). Semua bentuk v5 diawali
# Jalan/Gang/Perumahan/Kampung/..., sehingga "di fatmawati jakarta" (demo 22 Sep) lolos.
# Daerah di test_v4 (Fatmawati, Kemang, Margonda, Tebet) sengaja tidak dimasukkan.
BARE_AREAS = ["Cilandak", "Pondok Indah", "Rawamangun", "Kelapa Gading", "Cibubur", "Sunter",
              "Pasar Minggu", "Bintaro", "Serpong", "Antapani", "Kenjeran", "Darmo",
              "Tlogosari", "Condongcatur", "Seturan", "Jatiwaringin", "Pamulang", "Cengkareng",
              "Grogol", "Kalibata", "Pejaten", "Duren Sawit", "Buah Batu", "Ujung Berung",
              "Panakkukang", "Sukarame", "Banyumanik", "Gubeng", "Kotabaru", "Sei Sikambing"]
AREA_SUFFIX = ["", "", "", " Raya", " Dalam", " Selatan", " Barat", " Timur", " Utara", " Baru"]
BARE_RATIO = 0.15
# v7: arah mata angin menempel ke kota ("depok timur"). Kota/wilayah SAJA tetap bukan
# alamat ("gangguan di Bandung Utara"), jadi slot {C} juga diberi arah.
DIRECTIONS = ["Timur", "Barat", "Selatan", "Utara"]
BARE_AREAS_V7 = ["Jatiasih", "Cakung", "Ciledug", "Pasar Rebo", "Kebayoran Lama", "Tanjung Priok",
                 "Pademangan", "Cipinang", "Sukmajaya", "Beji", "Tajur", "Ciomas", "Arcamanik",
                 "Cicaheum", "Wonokromo", "Sukolilo", "Pedurungan", "Ngaliyan", "Umbulharjo",
                 "Medan Johor", "Helvetia", "Rappocini", "Biringkanaya", "Kemiling", "Rajabasa"]
V7 = True   # diubah oleh --v6 untuk membuat ulang data v6 (pembanding)
# v8: nama hari & bulan. Model v7 menandainya PERSON (Sabtu, Rabu, Agustus) karena data
# latih tidak pernah memuatnya — huruf kapital di tengah kalimat = "bentuk nama" baginya.
DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
          "September", "Oktober", "November", "Desember"]
V8 = True


def rand_name(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.2:
        return rng.choice(FIRST_NAMES)                     # "Budi"
    if r < 0.75:
        return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    if r < 0.93:
        return f"{rng.choice(FIRST_NAMES)} {rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    return f"{rng.choice(FIRST_NAMES)} br {rng.choice(BATAK_MARGA)}"   # "Rina br Ginting"


def rt_rw(rng: random.Random) -> str:
    rt = f"RT {rng.randint(1, 15):02d}"
    return f"{rt} RW {rng.randint(1, 12):02d}" if rng.random() < 0.6 else rt


def rand_address(rng: random.Random) -> str:
    kind = rng.random()
    no = rng.choice([f"No. {rng.randint(1, 250)}", f"No {rng.randint(1, 250)}",
                     str(rng.randint(1, 250))])   # "Jalan X 30 Medan" juga lazim
    city = rng.choice(CITIES)
    sep = ", " if rng.random() < 0.5 else " "
    street = f"{rng.choice(STREET_PREFIX)} {rng.choice(STREET_NAMES)}"
    if rng.random() < BARE_RATIO:                    # v6: tanpa awalan
        pool = BARE_AREAS + STREET_NAMES[:12] + (BARE_AREAS_V7 if V7 else [])
        area = rng.choice(pool) + rng.choice(AREA_SUFFIX)
        if V7 and rng.random() < 0.3 and " " not in city:
            city += " " + rng.choice(DIRECTIONS)
        return f"{area}{sep}{city}"
    if kind < 0.30:
        parts = [street]
        if rng.random() < 0.6:
            parts.append(no)
        if rng.random() < 0.8:
            parts.append(city)
        return sep.join(parts)
    if kind < 0.40:
        return f"{street} Km {rng.randint(1, 40)}{sep}{city}"
    if kind < 0.52:
        return (f"{rng.choice(KOMPLEKS)} Blok {rng.choice('ABCDEFGH')}{rng.randint(1, 20)} "
                f"{no}{sep}{city}")
    if kind < 0.64:
        return (f"{rng.choice(GANG)} {rt_rw(rng)}, "
                f"Kelurahan {rng.choice(KELURAHAN)}, {city}")
    if kind < 0.76:
        return f"{rng.choice(KAMPUNG)} {rt_rw(rng)}{sep}{city}"
    if kind < 0.86:
        return f"{rng.choice(DESA)} Kecamatan {rng.choice(KECAMATAN)}{sep}{city}"
    if kind < 0.94:
        return f"{rng.choice(DUSUN)} {rt_rw(rng)}{sep}{city}"
    return f"{street} {city}"


# {P} = PERSON, {A} = ADDRESS, {C} = nama kota TANPA label (kota saja bukan alamat).
# Termasuk kalimat TANPA entity dan kalimat berisi placeholder hasil regex, supaya
# model belajar bahwa itu BUKAN nama/alamat.
TEMPLATES = [
    "Nama saya {P} dan tinggal di {A}",
    "Nama saya {P}, alamat saya di {A}",
    "Halo kak, saya {P}. Internet di rumah saya {A} mati sejak kemarin",
    "Saya {P} mau komplain, wifi di {A} lemot banget",
    "Atas nama {P}, alamat pemasangan {A}",
    "Pelanggan atas nama {P} minta teknisi datang ke {A}",
    "Tolong kirim teknisi ke {A}, atas nama {P}",
    "Alamat saya {A}, nama {P}",
    "Saya ingin pindah alamat ke {A}",
    "Alamat baru saya {A}",
    "Tagihan atas nama {P} belum masuk",
    "Mohon bantuannya, saya {P}",
    "Dengan {P}, ada yang bisa dibantu?",
    "Saya {P}, nomor HP saya [REDACT_PHONE]",
    "Nama {P} NIK [REDACT_NIK] alamat {A}",
    "Kirim invoice ke [REDACT_EMAIL] atas nama {P}",
    "Rumah saya di {A}, sinyal hilang terus",
    "Teknisi belum datang ke {A} padahal sudah janji",
    "Perkenalkan saya {P} dari {A}",
    "Bisa dicek pemasangan baru di {A}?",
    # v4: gaya chat informal & alamat yang langsung disambung keluhan
    "aku {P}, mau lapor internet mati",
    "rumahku di {A}, wifinya gak nyala",
    "kak, {A} sinyalnya putus-putus",
    "{A} internetnya mati total dari semalam",
    "{P} di sini, modem saya bunyi terus",
    "Saya {P} dari {A}, mau upgrade paket",
    "Alamat pemasangan: {A}",
    "Nama pemilik akun mau diubah menjadi {P}",
    "Pemasangan di {A} kapan ya kak?",
    "Selamat siang, {P} mau tanya soal tagihan",
    # Negatif
    "Mau tanya tagihan bulan ini berapa ya",
    "Internet saya mati sejak kemarin sore",
    "Kenapa kuota saya cepat habis?",
    "Tolong hubungi saya di [REDACT_PHONE]",
    "Terima kasih kak atas bantuannya",
    "Saya mau upgrade paket internet ke 50 Mbps",
    "Jaringan di {C} sedang gangguan ya?",
    "Sinyal di daerah {C} sering hilang",
    "Sudah bayar lewat Bank Mandiri tapi belum masuk",
    "Oke siap kak",
    "Kapan teknisinya datang?",
    "Sudah saya restart modemnya kak",
    # Negatif — ditambahkan di v2 setelah demo: model sempat menandai "Oke" sebagai
    # PERSON dan "yang mana ya" sebagai ADDRESS. Kalimat demo itu sendiri TIDAK disalin.
    "Oke kak, alamat saya sudah benar belum di sistem?",
    "Siap, nama saya sudah terdaftar kan?",
    "Baik kak, alamat saya tidak berubah kok",
    "Oke, nama saya yang tercatat apa ya?",
    "Iya kak, alamat saya masih sama seperti dulu",
    "Sip, terima kasih, nama saya sudah saya kirim tadi",
    "Alamat saya yang lama atau yang baru ya kak?",
]


# v5: demo menemukan "aku lagi pusing sama pola pikir orang" disensor jadi 2 nama.
# Sebabnya: di v4 SETIAP "aku/saya + kata" di data latih diikuti nama, jadi model
# belajar "kata sesudah aku = nama". Obatnya kalimat curhat/keluhan tanpa PII yang
# dirakit dari subjek x keterangan x keadaan. Kata-kata di test_v3.jsonl sengaja TIDAK
# dimasukkan, supaya test_v3 mengukur generalisasi ke kata yang belum pernah dilihat.
SUBJECTS = ["aku", "saya", "gue", "gw", "sy", "aq", "Aku", "Saya", "Gue"]
ADVERBS = ["", "", "lagi", "sedang", "udah", "sudah", "masih", "baru", "juga", "jadi",
           "belum", "nggak", "gak", "tidak", "agak", "sangat", "bener-bener", "makin",
           "sempat", "pernah", "sering", "kadang"]
STATES = [
    "sedih", "marah", "bete", "galau", "lelah", "ngantuk", "lapar", "sibuk", "bahagia",
    "puas", "lega", "khawatir", "cemas", "panik", "kaget", "bosan", "jenuh", "sakit",
    "demam", "repot", "sabar", "ragu", "yakin", "setuju", "ngerti", "sungkan", "malu",
    "sewot", "emosi", "gemas", "penasaran", "curiga", "keberatan", "trauma", "bangga",
    "lemas", "pegal", "gerah", "kedinginan", "kepanasan", "kewalahan", "ketinggalan",
    "keheranan", "semangat", "optimis", "pesimis", "frustrasi", "dongkol", "jengkel",
    "mumet", "puyeng", "ribut", "tenang", "santai",
]
ACTIVITIES = [
    "mau tanya", "mau komplain", "mau lapor", "ingin berhenti", "butuh bantuan",
    "lupa password", "lupa bayar", "pakai paket 30 Mbps", "main game", "nonton film",
    "kuliah online", "meeting", "rapat", "belajar", "masak", "tidur", "ngantor",
    "lembur", "mudik", "liburan", "pulang kampung", "ngopi", "makan", "olahraga",
    "nyetir", "ngajar", "jualan online", "buka toko", "sekolah online", "streaming",
    "download file", "upload tugas", "video call", "baca email", "cek aplikasi",
    "ganti password", "restart router", "matikan modem", "nyalain ulang modem",
    "pindah kos", "renovasi rumah", "di jalan", "di kampus", "di sekolah", "di rumah",
    "di kos", "di pasar", "di bengkel", "di rumah sakit", "ngurus anak", "jaga toko",
    "ngerjain skripsi", "bikin laporan", "ngitung pengeluaran", "mikir ulang",
    "nyari solusi", "ngecek lampu modem", "ganti kabel LAN", "pasang repeater",
]
OBJECTS = ["sama tetangga", "sama bos", "sama anak-anak", "sama keluarga", "sama teman",
           "sama jaringan di sini", "sama tagihan bulan ini", "sama pelayanan teknisi",
           "sama aplikasinya", "sama harga paket", "sama kecepatan internet",
           "sama modem baru", "sama urusan kantor", "sama jadwal pemasangan"]
TAILS = ["", "", "", " kak", " min", " banget", " terus", " dari tadi pagi",
         ", internetnya malah mati", ", wifi putus-putus", ", tolong dibantu",
         " soalnya sinyal hilang", " gara-gara modem", ", kenapa ya", " nih",
         ", jaringannya lemot", " seharian"]
COMPLAINTS = ["internet mati", "wifi lemot", "modem mati", "mau cek tagihan",
              "teknisi belum datang", "sinyal hilang", "mau ganti paket", "lampu LOS merah"]
BANKS = ["BCA", "BRI", "BNI", "Mandiri", "minimarket", "e-wallet", "m-banking",
         "kantor pos", "teller bank"]


def rupiah(rng: random.Random) -> str:
    n = rng.randint(50, 2500) * 1000
    dotted = f"{n:,}".replace(",", ".")
    return rng.choice([f"Rp{dotted}", f"Rp {dotted}", f"Rp{n}", f"IDR {dotted}",
                       f"Rp. {dotted}", f"{n // 1000}rb", f"{n // 1000} ribu",
                       f"{dotted} rupiah"])


def subject_clause(rng: random.Random) -> str:
    """'aku lagi galau sama bos' — subjek + keadaan/aktivitas, tanpa PII."""
    adv = rng.choice(ADVERBS)
    if rng.random() < 0.5:
        pred = rng.choice(STATES)
        if rng.random() < 0.4:
            pred += " " + rng.choice(OBJECTS)
    else:
        pred = rng.choice(ACTIVITIES)
    return " ".join(w for w in (rng.choice(SUBJECTS), adv, pred) if w)


def build_composed(rng: random.Random) -> dict:
    """Kalimat rakitan: negatif curhat, nominal rupiah, atau curhat + nama asli."""
    r = rng.random()
    if r < 0.55:                                      # negatif murni
        text = subject_clause(rng) + rng.choice(TAILS)
        if rng.random() < 0.25:
            text += ", " + subject_clause(rng)
        return {"text": text, "entities": []}
    if r < 0.70:                                      # nominal rupiah, tanpa PII
        text = rng.choice([
            "Tagihan saya {R} padahal biasanya {R}", "sudah bayar {R} lewat {B}",
            "kok tagihannya jadi {R} ya", "{S}, tagihan {R} belum lunas?",
            "Mohon cek pembayaran {R} via {B}", "paket {R} per bulan dapat berapa Mbps?",
        ])
        text = (text.replace("{S}", subject_clause(rng), 1)
                    .replace("{B}", rng.choice(BANKS), 1))
        while "{R}" in text:
            text = text.replace("{R}", rupiah(rng), 1)
        return {"text": text, "entities": []}
    # Positif: curhat dulu, baru nama — supaya model tidak belajar "sesudah curhat
    # pasti bukan nama". Nama yang lolos = bocor, jadi pola ini wajib ada.
    name = rand_name(rng)
    head = subject_clause(rng) + rng.choice([", ", " ", ". "])
    mid = rng.choice(["aku ", "saya ", "nama saya ", "atas nama ", "ini ", "dengan "])
    tail = rng.choice([", ", " "]) + rng.choice(COMPLAINTS)
    if rng.random() < 0.3:
        tail += f", tagihan {rupiah(rng)}"
    start = len(head) + len(mid)
    return {"text": head + mid + name + tail, "entities": [[start, start + len(name), "PERSON"]]}


# v7: nama di posisi yang belum ada di v6 (setelah "pelanggan", "buat", "punya bu",
# "an.", "pelanggan:", ujung kalimat) dan alamat yang langsung disambung pertanyaan —
# model v6 kelebihan batas ("fatmawati jakarta, apakah") dan melewatkan nama ujung
# kalimat. Kalimat val/test TIDAK disalin; kata-katanya dibuat berbeda.
TEMPLATES_V7 = [
    "mohon dicek tunggakan pelanggan {P}",
    "Tunggakan atas pelanggan {P} berapa ya?",
    "rincian pemakaian bulan lalu buat {P} dong",
    "bisa kirim invoice buat {P}?",
    "laporan gangguan punya bu {P} gimana kelanjutannya?",
    "Pengaduan milik pak {P} belum direspon",
    "Tagihan a.n. {P} sudah dibayar kemarin",
    "tolong aktifkan lagi layanan an {P}",
    "Nama pelanggan: {P}. Keluhan: wifi mati",
    "pemilik akun: {P}, minta reset password",
    "Terdaftar dengan nama {P} sejak 2019",
    "tolong dibantu bapak {P} yang modemnya rusak",
    "Mohon dibantu ibu {P}, jaringannya lemot",
    "Kami mewakili keluarga {P}, mau berhenti langganan",
    "pasang baru di {A}, apakah masih ada slot minggu ini?",
    "Rumah saya di {A}, kenapa sinyal selalu hilang?",
    "saya pindah ke {A}, gimana prosedurnya?",
    "teknisi bisa ke {A}? kapan ya",
    "alamat {A}, apakah jaringan fiber sudah masuk?",
    "di {A} lagi gangguan, kapan normal?",
    "Mohon kirim teknisi ke {A}, terima kasih",
    "Gangguan di {C} sudah beres belum?",
    "apakah wilayah {C} sudah tercover fiber?",
    "Area {C} kok sering putus ya",
]

# v8: hari & bulan, dalam kalimat tanpa PII MAUPUN bersama nama/alamat — supaya model tidak
# belajar "kata berkapital di posisi ini = nama" dan tidak pula "ada hari = tidak ada nama".
TEMPLATES_V8 = [
    "Teknisi bisa datang hari {D} pagi tidak?",
    "jadwal pemasangan diundur ke {D} depan ya",
    "tagihan bulan {M} belum saya terima",
    "Sejak {M} internet sering putus",
    "{D} kemarin saya sudah lapor tapi belum ada kabar",
    "bisa dijadwalkan {D} siang?",
    "Pembayaran {M} dan {M} sudah lunas kan?",
    "layanan mati total sejak {D} malam",
    "mulai {M} tarif paket saya naik",
    "{M} lalu saya sempat berhenti langganan",
    "Teknisi datang hari {D} ke rumah {P} ya",
    "tagihan {M} atas nama {P} belum lunas",
    "Jadwalkan {D} pagi ke {A}",
    "sejak {M} saya tinggal di {A}",
    "{D} ini bisa kirim teknisi ke {A}?",
    "pelanggan {P} minta jadwal ulang hari {D}",
    "Pemasangan {M} atas nama {P} sudah dijadwalkan?",
    "{D} lalu teknisi ke {A} tapi rumah kosong",
]


def build_one(rng: random.Random) -> dict:
    if rng.random() < COMPOSED_RATIO:
        row = build_composed(rng)
        if rng.random() < LOWERCASE_RATIO:
            row["text"] = row["text"].lower()
        return row
    template = rng.choice(TEMPLATES + (TEMPLATES_V7 if V7 else []) + (TEMPLATES_V8 if V8 else []))
    text, entities, i = "", [], 0
    while i < len(template):
        slot = template[i:i + 3]
        if slot in ("{P}", "{A}", "{C}", "{D}", "{M}"):
            if slot == "{D}":
                text += rng.choice(DAYS)
            elif slot == "{M}":
                text += rng.choice(MONTHS)
            elif slot == "{C}":
                city = rng.choice(CITIES)
                if V7 and rng.random() < 0.3 and " " not in city:
                    city += " " + rng.choice(DIRECTIONS)
                text += city
            else:
                label = "PERSON" if slot == "{P}" else "ADDRESS"
                value = rand_name(rng) if label == "PERSON" else rand_address(rng)
                entities.append([len(text), len(text) + len(value), label])
                text += value
            i += 3
        else:
            text += template[i]
            i += 1
    # Panjang teks tidak berubah saat di-lowercase, jadi offset entity tetap valid.
    if rng.random() < LOWERCASE_RATIO:
        text = text.lower()
    return {"text": text, "entities": entities}


def held_out_tokens() -> set[str]:
    """Kata dari entity PERSON di val/test — tidak boleh ada di daftar nama generator."""
    toks = set()
    for fname in ("val.jsonl", "test_v2.jsonl", "test_v3.jsonl", "test_v4.jsonl", "test_v5.jsonl"):   # test v1 sudah terkontaminasi
        for line in open(Path(__file__).parent / fname, encoding="utf-8"):
            row = json.loads(line)
            for ent in row["entities"]:
                if ent[-1] == "PERSON":
                    value = ent[0] if len(ent) == 2 else row["text"][ent[0]:ent[1]]
                    toks.update(value.lower().split())
    return toks


def main() -> None:
    global V7, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", action="store_true", help="buat ulang data v6 (tanpa tambahan v7/v8)")
    ap.add_argument("--v7", action="store_true", help="buat ulang data v7 (tanpa tambahan v8)")
    ap.add_argument("--n", type=int, default=N_SENTENCES)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    global V8
    V7, V8, OUT_DIR = not args.v6, not (args.v6 or args.v7), Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leaked = held_out_tokens() & {n.lower() for n in FIRST_NAMES + LAST_NAMES}
    if leaked:
        raise SystemExit(f"nama val/test ada di daftar generator: {sorted(leaked)}")
    rng = random.Random(SEED)
    rows = [build_one(rng) for _ in range(args.n)]
    n_dev = int(len(rows) * DEV_RATIO)
    for name, part in (("dev.jsonl", rows[:n_dev]), ("train.jsonl", rows[n_dev:])):
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} kalimat")


if __name__ == "__main__":
    main()
