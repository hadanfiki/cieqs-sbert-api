from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Memuat model SBERT
model = SentenceTransformer("all-MiniLM-L6-v2")


# Data menu Cieqs Coffee
menu = [
    "Kopi Cieqs. Kopi dengan campuran gula aren. Kategori Coffee.",
    "Butterscotch. Kopi dengan rasa Butterscotch. Kategori Coffee.",
    "Kopi Susu. Kopi dengan campuran susu kental manis. Kategori Coffee."
]


# Membuat embedding menggunakan SBERT
embedding = model.encode(menu)


# Menghitung cosine similarity
similarity = cosine_similarity(embedding)


# Nama menu
nama_menu = [
    "Kopi Cieqs",
    "Butterscotch",
    "Kopi Susu"
]


print("\n=== HASIL COSINE SIMILARITY ===")

for i in range(len(nama_menu)):

    print(f"\nMenu: {nama_menu[i]}")

    for j in range(len(nama_menu)):

        if i != j:

            print(
                f"  Mirip dengan {nama_menu[j]}: "
                f"{similarity[i][j]:.4f}"
            )


# Ranking rekomendasi untuk Kopi Cieqs
target = 0

hasil_ranking = []

for i in range(len(nama_menu)):

    if i != target:

        hasil_ranking.append(
            (nama_menu[i], similarity[target][i])
        )


hasil_ranking.sort(
    key=lambda x: x[1],
    reverse=True
)


print("\n=== RANKING REKOMENDASI ===")

for nomor, (nama, nilai) in enumerate(
    hasil_ranking,
    start=1
):

    print(
        f"{nomor}. {nama} "
        f"(Similarity: {nilai:.4f})"
    )