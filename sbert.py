from sentence_transformers import SentenceTransformer

# Memuat model SBERT
model = SentenceTransformer("all-MiniLM-L6-v2")

# Data menu Cieqs Coffee
teks_menu = [
    "Kopi Cieqs. Kopi dengan campuran gula aren. Coffee",
    "Butterscotch. Minuman dengan rasa butterscotch. Coffee",
    "Kopi Susu. Kopi dengan campuran susu. Coffee"
]

# Membuat embedding
embedding = model.encode(teks_menu)

# Menampilkan hasil embedding
for i, vector in enumerate(embedding):
    print(f"\nEmbedding menu {i + 1}:")
    print(vector)