import requests

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# 1. MENGAMBIL DATA FIREBASE
# ==============================

PROJECT_ID = "cieqs-coffee"

url = (
    f"https://firestore.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/databases/(default)/documents/menus"
)

response = requests.get(url)

if response.status_code != 200:

    print("Gagal mengambil data Firebase.")
    print("Status:", response.status_code)
    exit()


data = response.json()


# ==============================
# 2. MEMBACA DATA MENU
# ==============================

nama_menu = []
teks_menu = []


for dokumen in data.get("documents", []):

    fields = dokumen.get("fields", {})

    nama = fields.get(
        "nama", {}
    ).get("stringValue", "")

    deskripsi = fields.get(
        "deskripsi", {}
    ).get("stringValue", "")

    kategori = fields.get(
        "kategori", {}
    ).get("stringValue", "")

    tersedia = fields.get(
        "tersedia", {}
    ).get("booleanValue", False)


    # Hanya menu yang tersedia
    if tersedia:

        nama_menu.append(nama)

        teks = (
            f"{nama}. "
            f"{deskripsi}. "
            f"Kategori {kategori}."
        )

        teks_menu.append(teks)


# ==============================
# 3. MENAMPILKAN DATA
# ==============================

print("\n=== DATA MENU DARI FIREBASE ===\n")

for i in range(len(nama_menu)):

    print(f"{i + 1}. {teks_menu[i]}")


# ==============================
# 4. MEMUAT MODEL SBERT
# ==============================

print("\nMemuat model SBERT...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==============================
# 5. MEMBUAT EMBEDDING
# ==============================

embedding = model.encode(
    teks_menu
)


# ==============================
# 6. COSINE SIMILARITY
# ==============================

similarity = cosine_similarity(
    embedding
)


# ==============================
# 7. MENAMPILKAN SIMILARITY
# ==============================

print("\n=== HASIL COSINE SIMILARITY ===")

for i in range(len(nama_menu)):

    print(f"\nMenu: {nama_menu[i]}")

    for j in range(len(nama_menu)):

        if i != j:

            print(
                f"  Mirip dengan {nama_menu[j]}: "
                f"{similarity[i][j]:.4f}"
            )


# ==============================
# 8. RANKING REKOMENDASI
# ==============================

# Contoh:
# target = menu pertama
# yaitu Kopi Cieqs

target = 0

hasil_ranking = []


for i in range(len(nama_menu)):

    if i != target:

        hasil_ranking.append(
            (
                nama_menu[i],
                similarity[target][i]
            )
        )


# Urutkan dari similarity terbesar
hasil_ranking.sort(
    key=lambda x: x[1],
    reverse=True
)


# ==============================
# 9. MENAMPILKAN RANKING
# ==============================

print("\n=== RANKING REKOMENDASI ===")

print(
    f"\nRekomendasi untuk: "
    f"{nama_menu[target]}"
)

for nomor, (nama, nilai) in enumerate(
    hasil_ranking,
    start=1
):

    print(
        f"{nomor}. {nama} "
        f"(Similarity: {nilai:.4f})"
    )