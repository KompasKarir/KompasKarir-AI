# 🧭 KompasKarir AI

> **Sistem Rekomendasi Program Studi Berbasis RIASEC & Kecerdasan Buatan**

REST API yang membantu siswa menemukan program studi paling cocok berdasarkan kepribadian RIASEC dan kemampuan akademik mereka — didukung TensorFlow, Cosine Similarity, dan narasi personal dari Gemini AI.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Gemini AI](https://img.shields.io/badge/Gemini_AI-Generative-4285F4?style=flat-square&logo=google)](https://ai.google.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://docker.com)

---

## 📌 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Fitur Utama](#-fitur-utama)
- [Cara Kerja FastAPI](#-cara-kerja-fastapi)
- [Integrasi Gemini AI](#-integrasi-gemini-ai)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Cara Menjalankan](#-cara-menjalankan-secara-lokal)
- [Menjalankan dengan Docker](#-menjalankan-dengan-docker)
- [API Endpoint](#-api-endpoint)
- [Struktur Project](#-struktur-project)
- [Environment Variables](#-environment-variables)
- [Tech Stack](#-tech-stack)
- [Troubleshooting](#-troubleshooting)

---

## 📖 Tentang Proyek

KompasKarir AI adalah backend API yang dirancang untuk membantu siswa SMA/SMK menentukan program studi yang paling sesuai dengan kepribadian dan kemampuan akademik mereka.

Sistem ini bekerja dengan cara:
1. Menerima **skor RIASEC** (Realistic, Investigative, Artistic, Social, Enterprising, Conventional) dan **skor akademik** pengguna
2. Menganalisis profil menggunakan **model Machine Learning** untuk memprediksi rumpun ilmu yang cocok
3. Merekomendasikan **program studi spesifik** menggunakan cosine similarity
4. Menghasilkan **narasi personal** yang hangat dan informatif menggunakan **Gemini AI**

---

## ✨ Fitur Utama

| Fitur | Keterangan |
|-------|------------|
| 🎯 Prediksi Rumpun Ilmu | Model Keras dengan layer perhatian kustom (`RIASECAttentionLayer`) mengklasifikasikan profil RIASEC ke rumpun ilmu yang paling sesuai |
| 📊 Rekomendasi Program Studi | Cosine similarity antara profil pengguna vs profil rata-rata tiap program studi |
| 🤖 Narasi Personal AI | Gemini AI menghasilkan ringkasan kepribadian, kekuatan, alasan kecocokan, dan saran pengembangan dalam Bahasa Indonesia |
| ⚡ SSE Streaming | Hasil prediksi dikirim bertahap via Server-Sent Events — UI bisa menampilkan hasil sebelum semua proses selesai |
| 🛡️ Fallback Otomatis | Jika Gemini AI tidak tersedia, narasi dibangun secara programatik dari data prediksi |
| 🔐 API Key Auth | Setiap request ke endpoint utama divalidasi menggunakan header `X-Internal-API-Key` |

---

## ⚡ Cara Kerja FastAPI

### Apa itu FastAPI?

**FastAPI** adalah web framework modern untuk Python yang digunakan untuk membangun REST API dengan cepat dan mudah. Di proyek ini, FastAPI berfungsi sebagai **pintu masuk** semua request dari frontend ke sistem AI.

### Kenapa Menggunakan FastAPI?

- **Cepat** — performa setara NodeJS dan Go
- **Validasi otomatis** — input request divalidasi otomatis menggunakan Pydantic
- **Dokumentasi otomatis** — tersedia di `/docs` tanpa konfigurasi tambahan
- **Async support** — mendukung pemrosesan asinkron untuk streaming SSE

### Bagaimana FastAPI Bekerja di Proyek Ini?

```
Client (Frontend)
      │
      │  POST /api/analyze
      │  Header: X-Internal-API-Key
      │  Body: { riasec: {...}, akademik: {...} }
      │
      ▼
┌─────────────────────────────────────┐
│            FastAPI App              │
│                                     │
│  1. Terima request                  │
│  2. Validasi API Key (middleware)   │
│  3. Validasi format input (Pydantic)│
│  4. Proses ke ML Engine             │
│  5. Stream hasil via SSE            │
└─────────────────────────────────────┘
      │
      │  text/event-stream (SSE)
      │  event: prediction → data: {...}
      │  event: narasi     → data: {...}
      │  data: [DONE]
      │
      ▼
Client menerima hasil bertahap
```

### Endpoint yang Tersedia

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| `GET` | `/` | Informasi dasar API |
| `GET` | `/health` | Cek status server |
| `POST` | `/api/analyze` | Analisis RIASEC + rekomendasi prodi |

### Validasi Input dengan Pydantic

FastAPI menggunakan **Pydantic** untuk memastikan data yang masuk valid sebelum diproses:

```python
# Contoh skema validasi yang digunakan
{
  "riasec": {
    "r": 70,   # harus float, 0-100
    "i": 85,   # harus float, 0-100
    "a": 60,   # harus float, 0-100
    "s": 55,   # harus float, 0-100
    "e": 45,   # harus float, 0-100
    "c": 50    # harus float, 0-100
  },
  "akademik": {
    "logika": 80,    # harus float, 0-100
    "bahasa": 75,    # harus float, 0-100
    "sains": 85,     # harus float, 0-100
    "sosial": 60,    # harus float, 0-100
    "praktik": 65    # harus float, 0-100
  },
  "top_k": 3  # harus int, 1-10
}
```

Jika ada field yang tidak valid (misalnya nilai di luar 0-100), FastAPI otomatis mengembalikan error `422 Unprocessable Entity` dengan penjelasan field mana yang salah.

### Dokumentasi Interaktif (Swagger UI)

Setelah server berjalan, buka browser dan akses:

```
http://localhost:8000/docs
```

Di sini kamu bisa langsung mencoba semua endpoint tanpa perlu tools tambahan seperti Postman.

---

## 🤖 Integrasi Gemini AI

### Apa itu Gemini AI?

**Gemini AI** adalah model kecerdasan buatan generatif dari Google yang mampu memahami konteks dan menghasilkan teks yang natural. Di proyek ini, Gemini digunakan untuk mengubah **data prediksi yang kering** menjadi **narasi personal yang hangat dan mudah dipahami** oleh siswa.

### Kenapa Menggunakan Gemini AI?

Tanpa Gemini, hasil prediksi hanya berupa angka dan label:
```
Rumpun: Sains & Teknologi (78.4%)
Prodi: Teknik Informatika (48.2%)
```

Dengan Gemini, hasil menjadi narasi yang personal dan bermakna:
```
"Kamu memiliki kepribadian investigatif yang kuat dengan kemampuan logika
yang menonjol. Ini menunjukkan bahwa kamu adalah tipe pemikir analitis yang
senang memecahkan masalah kompleks. Teknik Informatika adalah pilihan yang
sangat tepat untukmu karena..."
```

### Bagaimana Gemini AI Bekerja di Proyek Ini?

```
Data Prediksi dari ML Model
         │
         │  (kode RIASEC, rumpun, prodi, probabilitas)
         ▼
┌─────────────────────────────────┐
│         Prompt Builder          │
│                                 │
│  Menyusun prompt berisi:        │
│  - Profil RIASEC pengguna       │
│  - Hasil prediksi rumpun        │
│  - Daftar prodi yang cocok      │
│  - Instruksi format output JSON │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│         Gemini API Call         │
│                                 │
│  Model: gemini-3-flash-preview  │
│  Mode: async (thread terpisah)  │
│  Output: JSON terstruktur       │
└──────────────┬──────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
  ✅ Berhasil     ❌ Gagal/Timeout
  Kirim narasi    Gunakan narasi
  dari Gemini     fallback otomatis
```

### Output Narasi dari Gemini

Gemini menghasilkan narasi dalam format JSON terstruktur:

```json
{
  "kode_riasec": "IAS",
  "ringkasan": "Kamu adalah tipe Investigatif-Artistik yang memiliki rasa ingin tahu tinggi dan kemampuan berpikir kreatif-analitis yang kuat...",
  "kekuatan": [
    "Kemampuan logika dan analisis sangat menonjol",
    "Minat eksplorasi dan penelitian tinggi",
    "Mampu berpikir kreatif dalam memecahkan masalah"
  ],
  "alasan_kecocokan": "Teknik Informatika sangat cocok untukmu karena bidang ini membutuhkan kombinasi kemampuan logika tinggi dan kreativitas dalam merancang solusi teknologi...",
  "saran_pengembangan": "Mulai bangun portofolio coding kamu sekarang. Ikuti kompetisi programming atau hackathon untuk mengasah kemampuan dan memperluas jaringan..."
}
```

### Fallback Jika Gemini Gagal

Jika Gemini API tidak bisa diakses (timeout, quota habis, dll), sistem **tidak akan error** — melainkan otomatis menggunakan narasi fallback yang dibangun dari data prediksi:

```
Gemini gagal → Fallback aktif → Narasi tetap dikirim ke client
```

Ini memastikan aplikasi tetap berjalan normal meskipun layanan Gemini sedang bermasalah.

### Cara Mendapatkan Gemini API Key

1. Buka **[Google AI Studio](https://aistudio.google.com)**
2. Login dengan akun Google
3. Klik **"Get API Key"** → **"Create API Key"**
4. Copy API key yang dihasilkan
5. Masukkan ke file `.env` sebagai `GEMINI_API_KEY`

---

## 🏗️ Arsitektur Sistem

```
Client (Frontend / Postman)
        │
        │  POST /api/analyze
        ▼
┌───────────────────────────────────────────────────┐
│                   FastAPI App                     │
│                                                   │
│  ┌─────────────────┐   ┌─────────────────────┐   │
│  │  Auth Middleware │   │  Pydantic Validator  │   │
│  │  X-Internal-Key  │──▶│  RIASEC + Akademik  │   │
│  └─────────────────┘   └──────────┬──────────┘   │
│                                   │               │
│              ┌────────────────────┘               │
│              │                                    │
│    ┌─────────▼──────────┐  ┌──────────────────┐  │
│    │   TF/Keras Model   │  │    Gemini API     │  │
│    │                    │  │  (async thread)   │  │
│    │  1. Scale input    │  │                   │  │
│    │  2. Prediksi rumpun│  │  Generate narasi  │  │
│    │  3. Cosine sim     │  │  personal dalam   │  │
│    │  4. Top-K prodi    │  │  Bahasa Indonesia │  │
│    └─────────┬──────────┘  └────────┬─────────┘  │
│              │                      │             │
└──────────────┼──────────────────────┼─────────────┘
               │                      │
               ▼                      ▼
         event: prediction       event: narasi
               │                      │
               └──────────┬───────────┘
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

> 💡 Proses install TensorFlow membutuhkan beberapa menit. Pastikan koneksi internet stabil.

### 4. Buat File `.env`

Buat file `.env` di root project (sejajar dengan `app.py`):

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

Buka dokumentasi interaktif Swagger UI: **http://localhost:8000/docs**

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

Health check untuk memastikan server berjalan normal.

**Response:**
```json
{
  "status": "ok",
  "service": "capstone-fastapi",
  "version": "4.1.0"
}
```

---

### `GET /`

Informasi dasar API beserta daftar endpoint yang tersedia.

---

### `POST /api/analyze` ⭐

Endpoint utama — menerima skor RIASEC & akademik, mengembalikan prediksi rumpun ilmu + rekomendasi prodi + narasi personal Gemini AI via SSE streaming.

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

**Penjelasan Field:**

| Field | Tipe | Validasi | Keterangan |
|-------|------|----------|------------|
| `riasec.r` | float | 0 – 100 | Realistic — suka kerja fisik/praktis |
| `riasec.i` | float | 0 – 100 | Investigative — suka analisis/penelitian |
| `riasec.a` | float | 0 – 100 | Artistic — suka kreativitas/seni |
| `riasec.s` | float | 0 – 100 | Social — suka membantu orang lain |
| `riasec.e` | float | 0 – 100 | Enterprising — suka memimpin/bisnis |
| `riasec.c` | float | 0 – 100 | Conventional — suka keteraturan/data |
| `akademik.logika` | float | 0 – 100 | Kemampuan logika & matematika |
| `akademik.bahasa` | float | 0 – 100 | Kemampuan bahasa & komunikasi |
| `akademik.sains` | float | 0 – 100 | Kemampuan sains & IPA |
| `akademik.sosial` | float | 0 – 100 | Kemampuan IPS & sosial |
| `akademik.praktik` | float | 0 – 100 | Kemampuan praktik & keterampilan |
| `top_k` | int | 1 – 10 | Jumlah rumpun yang ditampilkan (default: 3) |

**Response (SSE Stream):**

Response dikirim bertahap dalam 3 event:

**Event 1 — `prediction`** (dikirim pertama, langsung setelah ML selesai):
```
event: prediction
data: {
  "kode_riasec": "IAS",
  "prediksi_utama": "Rumpun Sains & Teknologi",
  "top_personality": [
    { "label": "Rumpun Sains & Teknologi", "probabilitas": 78.4 },
    { "label": "Rumpun Kesehatan", "probabilitas": 12.1 },
    { "label": "Rumpun Sosial & Humaniora", "probabilitas": 9.5 }
  ],
  "rekomendasi": [
    {
      "rumpun": "Rumpun Sains & Teknologi",
      "kecocokan_persen": 78.4,
      "prodi_tersedia": [
        { "program_name": "Teknik Informatika", "similarity_persen": 48.2 },
        { "program_name": "Ilmu Komputer", "similarity_persen": 31.5 },
        { "program_name": "Teknik Elektro", "similarity_persen": 20.3 }
      ]
    }
  ]
}
```

**Event 2 — `narasi`** (dikirim setelah Gemini AI selesai):
```
event: narasi
data: {
  "kode_riasec": "IAS",
  "ringkasan": "Kamu memiliki kepribadian investigatif yang kuat dengan sentuhan artistik...",
  "kekuatan": [
    "Kemampuan logika dan analisis sangat menonjol",
    "Minat eksplorasi dan penelitian tinggi"
  ],
  "alasan_kecocokan": "Teknik Informatika sangat cocok karena bidang ini membutuhkan...",
  "saran_pengembangan": "Mulai bangun portofolio coding kamu sekarang..."
}
```

**Event 3 — Selesai:**
```
data: [DONE]
```

**Status Code:**

| Kode | Kondisi |
|------|---------|
| `200` | Berhasil, SSE stream dimulai |
| `401` | API key tidak ada atau salah |
| `422` | Input tidak valid (nilai di luar range, field kosong, dll) |
| `503` | `INTERNAL_API_KEY` belum dikonfigurasi di server |

---

## 📁 Struktur Project

```
KompasKarir-API/
├── app.py                  # Aplikasi utama FastAPI
├── requirements.txt        # Dependensi Python
├── Dockerfile              # Konfigurasi Docker
├── .gitignore              # File yang diabaikan Git
├── .gitattributes          # Konfigurasi Git LFS
├── README.md               # Dokumentasi ini
└── model_artifacts/        # File model Machine Learning
    ├── model_riasec.keras      # Model Keras utama (TensorFlow)
    ├── label_encoder.pkl       # Encoder nama rumpun ilmu
    ├── scaler_riasec.pkl       # Normalisasi input RIASEC
    ├── scaler_akademik.pkl     # Normalisasi input akademik
    ├── model_info.json         # Metadata model (akurasi, versi)
    ├── rumpun_list.json        # Daftar semua rumpun ilmu
    ├── rumpun_to_prodi.json    # Mapping rumpun → program studi
    ├── prodi_riasec_mean.json  # Profil rata-rata RIASEC per prodi
    └── prodi_akademik_mean.json # Profil rata-rata akademik per prodi
```

---

## ⚙️ Environment Variables

| Variable | Default | Keterangan |
|----------|---------|------------|
| `INTERNAL_API_KEY` | — | **Wajib.** Kunci autentikasi untuk endpoint `/api/analyze` |
| `GEMINI_API_KEY` | — | **Wajib.** API key Google Gemini untuk generate narasi personal |
| `GEMINI_MODEL` | `google/gemini-3-flash-preview` | Nama model Gemini yang digunakan |
| `MODEL_DIR` | `model_artifacts` | Direktori penyimpanan file model ML |
| `ALLOWED_ORIGINS` | `*` | Origin CORS yang diizinkan (pisahkan dengan koma untuk multiple) |
| `LOG_LEVEL` | `INFO` | Level logging: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## 🛠️ Tech Stack

| Komponen | Library / Tools | Fungsi |
|----------|----------------|--------|
| Web Framework | FastAPI 0.115, Uvicorn | Routing, middleware, SSE streaming |
| Validasi Input | Pydantic v2, pydantic-settings | Validasi request body & env variables |
| Machine Learning | TensorFlow 2.20, Keras 3.13 | Model klasifikasi rumpun ilmu |
| Feature Engineering | scikit-learn 1.6, NumPy | Scaling input, cosine similarity |
| Generative AI | Gemini via OpenAI-compatible client | Narasi personal dalam Bahasa Indonesia |
| Containerisasi | Docker (python:3.12-slim) | Deploy ke Hugging Face & server lain |

---

## 🔧 Troubleshooting

| Error / Gejala | Penyebab | Solusi |
|----------------|----------|--------|
| `401 Unauthorized` | API key salah atau tidak ada | Pastikan header `X-Internal-API-Key` terisi dengan nilai yang sama persis dengan `INTERNAL_API_KEY` di `.env` |
| `503 Service Unavailable` | `INTERNAL_API_KEY` kosong di server | Tambahkan `INTERNAL_API_KEY` ke file `.env` atau Secrets di Hugging Face |
| Model tidak ditemukan | Folder `model_artifacts/` tidak ada atau tidak lengkap | Pastikan semua file model ada di direktori yang sesuai dengan nilai `MODEL_DIR` |
| Narasi Gemini gagal | `GEMINI_API_KEY` salah atau quota habis | Cek API key di Google AI Studio — fallback narasi otomatis akan aktif |
| Push ke GitHub ditolak | File model terlalu besar (>100 MB) | Setup Git LFS: `git lfs track '*.keras' '*.pkl'` |
| CORS error di browser | Domain frontend tidak diizinkan | Tambahkan domain frontend ke `ALLOWED_ORIGINS` di `.env` |
| TensorFlow import error | Versi Python atau TF tidak sesuai | Pastikan Python 3.12 dan jalankan `pip install -r requirements.txt` ulang |
| Stream SSE berhenti tiba-tiba | Timeout atau error Gemini | Cek log server — fallback narasi seharusnya menangani ini secara otomatis |

---

## 🔗 Tautan Terkait

- 🤗 [Hugging Face Space](https://huggingface.co/spaces/RidhoHamdani/kompaskarir-ai) — Demo live API
- 📊 [KompasKarir-DS](https://github.com/KompasKarir/KompasKarir-DS) — Repository Data Science & training model
- 🌐 [KompasKarir-app](https://github.com/KompasKarir/KompasKarir-app) — Repository Frontend TypeScript

---

## 📄 Lisensi

[MIT License](LICENSE)
