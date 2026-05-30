---
title: RIASEC FastAPI
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# RIASEC Fast API

Sistem Rekomendasi Jurusan Pendidikan Berbasis RIASEC untuk Mendukung Pengambilan Keputusan Pelajar di Indonesia.

## Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Status API |
| GET | `/health` | Health check (untuk UptimeRobot) |
| GET | `/docs` | Swagger UI |
| GET | `/model-info` | Info performa model |
| POST | `/predict` | Prediksi rekomendasi jurusan |

## Contoh Request `/predict`

```json
{
  "r": 50.0,
  "i": 52.0,
  "a": 51.0,
  "s": 49.0,
  "e": 50.5,
  "c": 48.0,
  "logika": 75.0,
  "bahasa": 74.0,
  "sains": 73.0,
  "sosial": 76.0,
  "praktik": 72.0,
  "top_k": 3
}
```

## Contoh Response

```json
{
  "kode_riasec": "IAS",
  "top_personality": [
    {"kode": "i", "skor": 52.0},
    {"kode": "a", "skor": 51.0},
    {"kode": "s", "skor": 49.0}
  ],
  "rekomendasi": [
    {
      "rumpun": "Sains & Teknologi",
      "kecocokan_persen": 45.23,
      "prodi_tersedia": [
        {"program_name": "Teknik Informatika", "similarity_persen": 92.1},
        {"program_name": "Sistem Informasi", "similarity_persen": 88.5},
        {"program_name": "Ilmu Komputer", "similarity_persen": 85.3}
      ]
    }
  ]
}
```
