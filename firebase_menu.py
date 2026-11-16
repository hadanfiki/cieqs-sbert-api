import requests


# ID project Firebase
PROJECT_ID = "cieqs-coffee"


# URL Firestore REST API
url = (
    f"https://firestore.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/databases/(default)/documents/menus"
)


# Mengambil data menu
response = requests.get(url)


if response.status_code == 200:

    data = response.json()

    print("\n=== DATA MENU DARI FIREBASE ===\n")

    for dokumen in data.get("documents", []):

        fields = dokumen.get("fields", {})

        nama = fields.get("nama", {}).get("stringValue", "")
        deskripsi = fields.get("deskripsi", {}).get("stringValue", "")
        kategori = fields.get("kategori", {}).get("stringValue", "")
        tersedia = fields.get("tersedia", {}).get("booleanValue", False)

        print(f"Nama       : {nama}")
        print(f"Deskripsi  : {deskripsi}")
        print(f"Kategori   : {kategori}")
        print(f"Tersedia   : {tersedia}")
        print("-" * 40)

else:

    print("Gagal mengambil data Firebase.")
    print("Status:", response.status_code)
    print(response.text)