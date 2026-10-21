
import os
import requests
from dotenv import load_dotenv

load_dotenv()

BLS_API_KEY = os.getenv("BLS_API_KEY")
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

def test_connection():
    if not BLS_API_KEY:
        print("BLS_API_KEY tidak ditemukan di .env")
        return

    payload = {
        "seriesid": ["CUUR0000SA0"],
        "startyear": "2025",
        "endyear": "2026",
        "registrationkey": BLS_API_KEY,
    }

    try:
        response = requests.post(
            BLS_API_URL,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        print("HTTP Status:", response.status_code)
        print("BLS Status:", result.get("status"))
        print("Pesan:", result.get("message"))

        if result.get("status") == "REQUEST_SUCCEEDED":
            print("Koneksi BLS API berhasil!")
            series = result.get("Results", {}).get("series", [])
            if series:
                data = series[0].get("data", [])
                print("Jumlah observasi:", len(data))
                print("Contoh data terbaru:", data[:3])
        else:
            print("Permintaan belum berhasil. Periksa pesan dari BLS.")

    except requests.RequestException as error:
        print("Gagal menghubungi BLS API:", error)


if __name__ == "__main__":
    test_connection()
