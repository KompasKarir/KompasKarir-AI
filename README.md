# 🧭 KompasKarir AI

> **Sistem Rekomendasi Program Studi Berbasis RIASEC & Kecerdasan Buatan**

REST API yang membantu siswa menemukan program studi paling cocok berdasarkan kepribadian RIASEC dan kemampuan akademik mereka — didukung TensorFlow, Cosine Similarity, dan narasi personal dari Gemini AI.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://docker.com)

---

## ✨ Fitur Utama

- **Prediksi Rumpun Ilmu** — model Keras dengan layer perhatian kustom (`RIASECAttentionLayer`) mengklasifikasikan profil RIASEC ke rumpun ilmu yang paling sesuai
- **Rekomendasi Program Studi** — cosine similarity antara profil pengguna vs rata-rata tiap prodi
- **Narasi Personal AI** — Gemini menghasilkan ringkasan, kekuatan, alasan kecocokan, dan saran pengembangan dalam Bahasa Indonesia
- **SSE Streaming** — hasil prediksi dikirim bertahap via Server-Sent Events, sehingga UI bisa tampil progresif
- **Fallback Otomatis** — jika Gemini tidak tersedia, narasi dibangun dari data prediksi secara programatik

---

## 🏗️ Arsitektur

```
Request (POST /api/analyze)
        │
        ▼
┌─────────────────────┐
│   Auth Middleware   │  ← X-Internal-API-Key header
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Pydantic Validator │  ← RIASEC (6 dim) + Akademik (5 dim)
└────────┬────────────┘
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
┌──────────────┐        ┌─────────────────┐
│  TF/Keras    │        │  Gemini API     │
│  Classifier  │        │  (async thread) │
│  + Cosine    │        │                 │
│  Similarity  │        │  Fallback jika  │
└──────┬───────┘        │  Gemini gagal   │
       │                └────────┬────────┘
       │                         │
       ▼                         ▼
  event: prediction         event: narasi
       │                         │
       └──────────┬──────────────┘
                  ▼
            data: [DONE]
         (SSE Stream selesai)
```

---

## 🚀 Cara Menjalankan Secara Lokal

### 1. Clone Repository

```bash
git clone https://github.com/KompasKarir/KompasKarir-API.git
cd KompasKarir-API
```

### 2. Buat Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Buat File `.env`

Buat file `.env` di root project:

```env
INTERNAL_API_KEY=rahasia-kunci-internal-kamu
GEMINI_API_KEY=api-key-gemini-kamu
GEMINI_MODEL=google/gemini-3-flash-preview
MODEL_DIR=model_artifacts
ALLOWED_ORIGINS=*
LOG_LEVEL=INFO
```

> ⚠️ **Jangan pernah commit file `.env` ke GitHub!** Pastikan sudah ada di `.gitignore`.

### 5. Jalankan Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Buka dokumentasi interaktif: **http://localhost:8000/docs**

---

## 🐳 Menjalankan dengan Docker

```bash
# Build image
docker build -t kompaskarir-ai .

# Jalankan container
docker run -p 7860:7860 \
  -e INTERNAL_API_KEY=kunci-kamu \
  -e GEMINI_API_KEY=key-gemini-kamu \
  kompaskarir-ai
```

---

## 📡 API Endpoint

### `GET /health`
Health check service.

**Response:**
```json
{ "status": "ok", "service": "capstone-fastapi", "version": "4.1.0" }
```

---

### `GET /`
Informasi dasar API.

---

### `POST /api/analyze` ⭐

Endpoint utama — menerima skor RIASEC & akademik, mengembalikan prediksi + narasi via SSE stream.

**Header yang diperlukan:**
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

| Field | Tipe | Validasi | Keterangan |
|-------|------|----------|------------|
| `riasec.r/i/a/s/e/c` | float | 0 – 100 | Skor dimensi kepribadian RIASEC |
| `akademik.logika/bahasa/sains/sosial/praktik` | float | 0 – 100 | Skor kemampuan akademik |
| `top_k` | int | 1 – 10, default 3 | Jumlah rumpun yang ditampilkan |

**Response (SSE Stream):**

```
event: prediction
data: {
  "kode_riasec": "IAS",
  "prediksi_utama": "Rumpun Sains & Teknologi",
  "top_personality": [
    { "label": "Rumpun Sains & Teknologi", "probabilitas": 78.4 },
    { "label": "Rumpun Kesehatan", "probabilitas": 12.1 }
  ],
  "rekomendasi": [
    {
      "rumpun": "Rumpun Sains & Teknologi",
      "kecocokan_persen": 78.4,
      "prodi_tersedia": [
        { "program_name": "Teknik Informatika", "similarity_persen": 48.2 },
        { "program_name": "Ilmu Komputer", "similarity_persen": 31.5 }
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

| Kode | Kondisi |
|------|---------|
| `200` | Berhasil, SSE stream dimulai |
| `401` | API key tidak ada atau salah |
| `422` | Input tidak valid |
| `503` | INTERNAL_API_KEY belum dikonfigurasi di server |

---

## 📁 Struktur Project

```
kompaskarir-ai/
├── app.py                  # Aplikasi utama FastAPI
├── requirements.txt        # Dependensi Python
├── Dockerfile              # Konfigurasi Docker
├── .gitignore
├── .gitattributes          # Konfigurasi Git LFS
├── README.md
└── model_artifacts/        # File model ML (tidak di-commit jika besar)
    ├── model_riasec.keras
    ├── label_encoder.pkl
    ├── scaler_riasec.pkl
    ├── scaler_akademik.pkl
    ├── model_info.json
    ├── rumpun_list.json
    ├── rumpun_to_prodi.json
    ├── prodi_riasec_mean.json
    └── prodi_akademik_mean.json
```

---

## ⚙️ Environment Variables

| Variable | Default | Keterangan |
|----------|---------|------------|
| `INTERNAL_API_KEY` | — | **Wajib.** Kunci autentikasi endpoint |
| `GEMINI_API_KEY` | — | **Wajib.** API key untuk narasi Gemini |
| `GEMINI_MODEL` | `google/gemini-3-flash-preview` | Nama model Gemini |
| `MODEL_DIR` | `model_artifacts` | Direktori model ML |
| `ALLOWED_ORIGINS` | `*` | Origin CORS yang diizinkan |
| `LOG_LEVEL` | `INFO` | Level logging |

---

## 🛠️ Tech Stack

| Komponen | Library / Tools |
|----------|----------------|
| Web Framework | FastAPI 0.115, Uvicorn |
| Validasi Input | Pydantic v2, pydantic-settings |
| Machine Learning | TensorFlow 2.20, Keras 3.13 |
| Feature Engineering | scikit-learn 1.6, NumPy |
| Generative AI | Gemini (OpenAI-compatible client) |
| Containerisasi | Docker (python:3.12-slim) |

---

## 🔧 Troubleshooting

| Error | Solusi |
|-------|--------|
| `401 Unauthorized` | Pastikan header `X-Internal-API-Key` terisi dan nilainya benar |
| `503 Service Unavailable` | `INTERNAL_API_KEY` kosong di server — tambahkan ke `.env` |
| Model tidak ditemukan | Pastikan folder `model_artifacts/` ada dan lengkap |
| Narasi Gemini gagal | Cek `GEMINI_API_KEY` — fallback narasi otomatis aktif |
| Push ditolak (file besar) | Setup Git LFS: `git lfs track '*.keras' '*.pkl'` |
| CORS error | Tambahkan domain frontend ke `ALLOWED_ORIGINS` di `.env` |

---

## 📄 Lisensi

[MIT License](LICENSE)
