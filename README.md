<div align="center">
  <img alt="KompasKarir AI" height="100" src="https://raw.githubusercontent.com/SirGhazian/kompas-karir/refs/heads/main/public/images/app-logo.png">
</div>

<img src="https://github.com/user-attachments/assets/d37a62f7-650a-4886-81c9-d3809d3ddeed" width="100%" height="2px"/>

Backend API sistem rekomendasi program studi berbasis kepribadian RIASEC dengan generative AI Gemini untuk narasi personal.

**Capstone Project** Coding Camp 2026 oleh Dicoding × DBS Foundation (Tim PSU305)

## Tentang

KompasKarir AI adalah backend berbasis FastAPI yang menjadi otak dari sistem rekomendasi program studi. API ini menerima profil RIASEC dan nilai akademik pengguna, kemudian menganalisisnya menggunakan model Machine Learning (TensorFlow) untuk memprediksi rumpun ilmu yang cocok dan merekomendasikan program studi spesifik. Hasilnya diperkaya dengan narasi personal dalam Bahasa Indonesia yang dihasilkan oleh Gemini AI.

Repo ini terdiri dari dua bagian utama:

- `training/` — notebook Google Colab untuk melatih model Machine Learning
- `app.py` — backend FastAPI yang melayani hasil prediksi model ke frontend

## Fitur

- **Prediksi Rumpun Ilmu** — model Keras dengan layer perhatian kustom (`RIASECAttentionLayer`) mengklasifikasikan profil RIASEC ke rumpun ilmu yang paling sesuai
- **Rekomendasi Program Studi** — cosine similarity antara profil pengguna vs profil rata-rata tiap program studi
- **Narasi Personal (Gemini AI)** — menghasilkan ringkasan kepribadian, kekuatan, alasan kecocokan, dan saran pengembangan dalam Bahasa Indonesia
- **SSE Streaming** — hasil dikirim bertahap via Server-Sent Events sehingga UI dapat menampilkan data sebelum semua proses selesai
- **Fallback Otomatis** — jika Gemini AI tidak tersedia, narasi dibangun secara programatik dari data prediksi

## Cara Kerja FastAPI

FastAPI berfungsi sebagai pintu masuk seluruh request dari frontend. Setiap request divalidasi secara otomatis menggunakan Pydantic sebelum diproses oleh sistem AI.

```
Request (POST /api/analyze)
        |
        v
Auth Middleware (X-Internal-API-Key)
        |
        v
Pydantic Validator (RIASEC + Akademik, range 0-100)
        |
        |-------------------|
        v                   v
TF/Keras Classifier     Gemini API (async)
+ Cosine Similarity     Narasi personal
        |                   |
        v                   v
  event: prediction    event: narasi
        |                   |
        |-------------------|
                  |
                  v
            data: [DONE]
```

Dokumentasi interaktif tersedia di `/docs` (Swagger UI) setelah server berjalan — bisa digunakan langsung tanpa tools tambahan seperti Postman.

## Cara Kerja Gemini AI

Gemini AI digunakan untuk mengubah data prediksi yang berupa angka dan label menjadi narasi yang personal dan mudah dipahami siswa.

Tanpa Gemini, hasil hanya berupa:

```
Rumpun: Sains & Teknologi (78.4%)
Prodi: Teknik Informatika (48.2%)
```

Dengan Gemini, hasil menjadi narasi bermakna:

```json
{
  "ringkasan": "Kamu memiliki kepribadian investigatif yang kuat...",
  "kekuatan": ["Kemampuan logika menonjol", "Minat sains tinggi"],
  "alasan_kecocokan": "Teknik Informatika cocok karena...",
  "saran_pengembangan": "Perkuat portofolio coding kamu..."
}
```

Gemini dipanggil secara async di thread terpisah sehingga tidak memblokir pengiriman hasil prediksi. Jika Gemini gagal, sistem otomatis menggunakan narasi fallback — aplikasi tetap berjalan normal.

## Cara Kerja Training Model

Kode training ada di folder `training/` dan dijalankan di Google Colab. Berikut alur lengkapnya:

| Tahap               | Proses                                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1. Load Dataset     | Upload file `riasec_ml_ready.csv` ke Colab                                                                        |
| 2. Eksplorasi       | Visualisasi distribusi data, skor RIASEC, dan korelasi fitur                                                      |
| 3. Preprocessing    | Encoding label, scaling fitur (StandardScaler), penyeimbangan data (SMOTE)                                        |
| 4. Split Dataset    | Train 70% / Validasi 15% / Test 15% dengan stratifikasi                                                           |
| 5. Arsitektur Model | Dual-input Keras: cabang RIASEC (dengan `RIASECAttentionLayer`) + cabang Akademik, digabung lalu diklasifikasikan |
| 6. Training         | Custom training loop dengan ReduceLROnPlateau, EarlyStopping, dan AccuracyThreshold                               |
| 7. Evaluasi         | Classification report, confusion matrix, top-3 accuracy, sharpened MAE                                            |
| 8. Simpan Artefak   | Model `.keras`, scaler `.pkl`, encoder `.pkl`, mapping JSON                                                       |
| 9. Download         | Semua artefak di-zip dan diunduh untuk digunakan di backend                                                       |

Hasil training (folder `model_artifacts/`) langsung dipakai oleh `app.py` saat melayani request.

## Tech Stack

| Komponen             | Teknologi                         |
| -------------------- | --------------------------------- |
| Web Framework        | FastAPI 0.115, Uvicorn            |
| Validasi Input       | Pydantic v2, pydantic-settings    |
| Machine Learning     | TensorFlow 2.20, Keras 3.13       |
| Feature Engineering  | scikit-learn 1.6, NumPy           |
| Generative AI        | Gemini (OpenAI-compatible client) |
| Containerisasi       | Docker (python:3.12-slim)         |
| Training Environment | Google Colab                      |

## Memulai

### Prasyarat

- Python 3.12+
- pip

### Instalasi

```bash
git clone https://github.com/KompasKarir/KompasKarir-AI.git
cd KompasKarir-AI
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Buat file `.env` di root project:

```env
INTERNAL_API_KEY=rahasia-kunci-internal
GEMINI_API_KEY=api-key-gemini
GEMINI_MODEL=google/gemini-3-flash-preview
MODEL_DIR=model_artifacts
ALLOWED_ORIGINS=*
LOG_LEVEL=INFO
```

> Jangan pernah commit file `.env` ke GitHub. Pastikan sudah ada di `.gitignore`.

Cara mendapatkan `GEMINI_API_KEY`: buka [Google AI Studio](https://aistudio.google.com) → Get API Key → Create API Key.

### Development

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Buka [http://localhost:8000/docs](http://localhost:8000/docs) untuk dokumentasi interaktif.

### Docker

```bash
docker build -t kompaskarir-ai .
docker run -p 7860:7860 \
  -e INTERNAL_API_KEY=kunci-kamu \
  -e GEMINI_API_KEY=key-gemini-kamu \
  kompaskarir-ai
```

## Melatih Ulang Model

> **Download model artifacts:** [Google Drive](https://drive.google.com/drive/folders/1gzyu7shSc-V4MWJGdZzPXDXUuJA_v6wr?usp=sharing)
> — Ekstrak dan letakkan folder `model_artifacts/` di root project untuk langsung menjalankan backend tanpa perlu training ulang.

1. Buka notebook di folder `training/` menggunakan Google Colab
2. Upload file dataset `riasec_ml_ready.csv` saat diminta
3. Jalankan semua cell secara berurutan
4. Di akhir notebook, file `model_artifacts.zip` akan otomatis terunduh
5. Ekstrak ZIP dan letakkan folder `model_artifacts/` di root project ini

## API Endpoint

### GET /health

Health check untuk memastikan server berjalan normal.

```json
{ "status": "ok", "service": "capstone-fastapi", "version": "4.1.0" }
```

### GET /

Informasi dasar API beserta daftar endpoint yang tersedia.

### POST /api/analyze

Endpoint utama — menerima profil RIASEC dan nilai akademik, mengembalikan prediksi rumpun ilmu, rekomendasi program studi, dan narasi personal Gemini AI via SSE streaming.

**Header:**

```
X-Internal-API-Key: <INTERNAL_API_KEY>
Content-Type: application/json
```

**Request Body:**

```json
{
  "riasec": {
    "r": 70,
    "i": 85,
    "a": 60,
    "s": 55,
    "e": 45,
    "c": 50
  },
  "akademik": {
    "logika": 80,
    "bahasa": 75,
    "sains": 85,
    "sosial": 60,
    "praktik": 65
  },
  "top_k": 3
}
```

| Field                                         | Tipe  | Validasi | Keterangan                                  |
| --------------------------------------------- | ----- | -------- | ------------------------------------------- |
| `riasec.r/i/a/s/e/c`                          | float | 0 – 100  | Skor tiap dimensi kepribadian RIASEC        |
| `akademik.logika/bahasa/sains/sosial/praktik` | float | 0 – 100  | Skor kemampuan akademik per bidang          |
| `top_k`                                       | int   | 1 – 10   | Jumlah rumpun yang ditampilkan (default: 3) |

**Response — SSE Stream (3 event berurutan):**

```
event: prediction
data: {
  "kode_riasec": "IAS",
  "prediksi_utama": "Rumpun Sains & Teknologi",
  "top_personality": [
    { "label": "Rumpun Sains & Teknologi", "probabilitas": 78.4 }
  ],
  "rekomendasi": [
    {
      "rumpun": "Rumpun Sains & Teknologi",
      "kecocokan_persen": 78.4,
      "prodi_tersedia": [
        { "program_name": "Teknik Informatika", "similarity_persen": 48.2 }
      ]
    }
  ]
}

event: narasi
data: {
  "kode_riasec": "IAS",
  "ringkasan": "Kamu memiliki minat investigatif yang kuat...",
  "kekuatan": ["Kemampuan logika menonjol", "Minat sains tinggi"],
  "alasan_kecocokan": "Teknik Informatika cocok karena...",
  "saran_pengembangan": "Perkuat portofolio coding kamu..."
}

data: [DONE]
```

**Status Code:**

| Kode  | Kondisi                                        |
| ----- | ---------------------------------------------- |
| `200` | Berhasil, SSE stream dimulai                   |
| `401` | API key tidak ada atau salah                   |
| `422` | Input tidak valid                              |
| `503` | INTERNAL_API_KEY belum dikonfigurasi di server |

## Struktur Project

```
KompasKarir-ai/
├── app.py                        # Aplikasi utama FastAPI
├── requirements.txt              # Dependensi Python
├── Dockerfile                    # Konfigurasi Docker
├── .gitignore
├── .gitattributes                # Konfigurasi Git LFS
├── README.md
├── training/                     # Kode training model (Google Colab)
│   └── training_riasec.ipynb    # Notebook training lengkap
└── model_artifacts/              # File model hasil training
    ├── model_riasec.keras        # Model Keras (TensorFlow)
    ├── label_encoder.pkl         # Encoder nama rumpun ilmu
    ├── scaler_riasec.pkl         # Normalisasi input RIASEC
    ├── scaler_akademik.pkl       # Normalisasi input akademik
    ├── model_info.json           # Metadata model (akurasi, versi)
    ├── rumpun_list.json          # Daftar rumpun ilmu
    ├── rumpun_to_prodi.json      # Mapping rumpun ke program studi
    ├── prodi_riasec_mean.json    # Profil rata-rata RIASEC per prodi
    └── prodi_akademik_mean.json  # Profil rata-rata akademik per prodi
```

## Troubleshooting

| Error                     | Solusi                                                              |
| ------------------------- | ------------------------------------------------------------------- |
| `401 Unauthorized`        | Pastikan header `X-Internal-API-Key` terisi dengan nilai yang benar |
| `503 Service Unavailable` | `INTERNAL_API_KEY` kosong — tambahkan ke `.env` atau Secrets HF     |
| Model tidak ditemukan     | Pastikan folder `model_artifacts/` ada dan semua file lengkap       |
| Narasi Gemini gagal       | Cek `GEMINI_API_KEY` — fallback narasi otomatis akan aktif          |
| Push GitHub ditolak       | Setup Git LFS: `git lfs track '*.keras' '*.pkl'`                    |
| CORS error                | Tambahkan domain frontend ke `ALLOWED_ORIGINS` di `.env`            |

## Tim Pengembang

| ID             | Nama                                  | Peran                    |
| -------------- | ------------------------------------- | ------------------------ |
| CDCC282D6Y1250 | Carli Tamba                           | Data Scientist           |
| CDCC012D6Y1245 | Muhammad Firman Ardiansyah            | Data Scientist           |
| CFCC282D6Y0786 | Muhammad Ghazian Tsaqif Zhafiri Andoz | Full-Stack Web Developer |
| CFCC955D6Y1821 | Dhimas Setyo Wahyu Santoso            | Full-Stack Web Developer |
| CACC282D6Y0961 | Ridho Hamdani Putra                   | AI Engineer              |
| CACC282D6X0960 | Setya Carina Rianti                   | AI Engineer              |

## Lisensi

Proyek ini dikembangkan sebagai bagian dari program Coding Camp 2026 (Dicoding × DBS Foundation).

## Links

| Deployment                         | URL                                                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Deployment Frontend (Vercel)       | [kompaskarir.vercel.app](https://kompaskarir.vercel.app)                                                |
| Deployment Backend (Hugging Face)  | [SirGhazian/kompaskarir-backend](https://huggingface.co/spaces/SirGhazian/kompaskarir-backend)          |
| Deployment AI Model (Hugging Face) | [RidhoHamdani/kompaskarir-ai](https://huggingface.co/spaces/RidhoHamdani/kompaskarir-ai)                |
| Model Artifacts (Google Drive)     | [model_artifacts](https://drive.google.com/drive/folders/1gzyu7shSc-V4MWJGdZzPXDXUuJA_v6wr?usp=sharing) |

| All Source Code          | URL                                           |
| ------------------------ | --------------------------------------------- |
| Main GitHub Organization | [KompasKarir](https://github.com/KompasKarir) |
