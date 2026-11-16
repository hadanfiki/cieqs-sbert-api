import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(
    page_title="Cieqs Coffee Recommendation",
    page_icon="☕"
)

st.title("☕ Cieqs Coffee")
st.subheader("Rekomendasi Menu dengan SBERT")

PROJECT_ID = "cieqs-coffee"

FIREBASE_URL = (
    f"https://firestore.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/databases/(default)/documents/menus"
)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


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


menu = ambil_menu()

if not menu:

    st.error("Data menu tidak ditemukan.")

else:

    pilihan = st.selectbox(
        "Pilih menu:",
        [item["nama"] for item in menu]
    )

    if st.button("Tampilkan Rekomendasi"):

        model = load_model()

        teks_menu = []

        for item in menu:

            teks = (
                f"{item['nama']}. "
                f"{item['deskripsi']}. "
                f"Kategori {item['kategori']}."
            )

            teks_menu.append(teks)

        embedding = model.encode(teks_menu)

        target_index = -1

        for i in range(len(menu)):

            if menu[i]["nama"].lower() == pilihan.lower():

                target_index = i
                break

        similarity = cosine_similarity(embedding)

        hasil = []

        for i in range(len(menu)):

            if i != target_index:

                hasil.append({
                    "nama": menu[i]["nama"],
                    "similarity": float(
                        similarity[target_index][i]
                    )
                })

        hasil.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        st.success(
            f"Rekomendasi untuk: {pilihan}"
        )

        for nomor, item in enumerate(
            hasil,
            start=1
        ):

            persentase = (
                item["similarity"] * 100
            )

            st.write(
                f"**{nomor}. {item['nama']}**"
            )

            st.write(
                f"Tingkat kemiripan: "
                f"{persentase:.2f}%"
            )
