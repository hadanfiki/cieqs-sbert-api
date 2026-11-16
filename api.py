from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
CORS(app)


# ==============================
# LOAD MODEL SBERT
# ==============================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================
# FIREBASE
# ==============================

PROJECT_ID = "cieqs-coffee"

FIREBASE_URL = (
    f"https://firestore.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/databases/(default)/documents/menus"
)


# ==============================
# FUNGSI AMBIL MENU FIREBASE
# ==============================

def ambil_menu():

    response = requests.get(FIREBASE_URL)

    if response.status_code != 200:
        return []

    data = response.json()

    menu = []

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

        if tersedia:

            menu.append({
                "nama": nama,
                "deskripsi": deskripsi,
                "kategori": kategori
            })

    return menu


# ==============================
# ENDPOINT REKOMENDASI
# ==============================

@app.route("/rekomendasi")
def rekomendasi():

    target = request.args.get("menu")

    if not target:
        return jsonify({
            "error": "Menu belum diberikan"
        })


    menu = ambil_menu()

    if not menu:
        return jsonify({
            "error": "Data menu tidak ditemukan"
        })


    # Membuat teks untuk SBERT
    teks_menu = []

    for item in menu:

        teks = (
            f"{item['nama']}. "
            f"{item['deskripsi']}. "
            f"Kategori {item['kategori']}."
        )

        teks_menu.append(teks)


    # Embedding
    embedding = model.encode(teks_menu)


    # Cari index menu target
    target_index = -1

    for i in range(len(menu)):

        if menu[i]["nama"].lower() == target.lower():

            target_index = i
            break


    if target_index == -1:

        return jsonify({
            "error": "Menu target tidak ditemukan"
        })


    # Cosine Similarity
    similarity = cosine_similarity(
        embedding
    )


    # Ranking
    hasil = []

    for i in range(len(menu)):

        if i != target_index:

            hasil.append({
                "nama": menu[i]["nama"],
                "similarity": round(
                    float(
                        similarity[target_index][i]
                    ),
                    4
                )
            })


    hasil.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )


    return jsonify({
        "menu": target,
        "rekomendasi": hasil
    })


# ==============================
# MENJALANKAN SERVER
# ==============================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )