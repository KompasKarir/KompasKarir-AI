import asyncio
import json
import logging
import os
import pickle
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Any, AsyncGenerator

import numpy as np
import tensorflow as tf
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Logging

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Settings

class Settings(BaseSettings):
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "model_artifacts")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Auth

api_key_header = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)


async def verify_internal_key(api_key: str = Security(api_key_header)) -> str:
    settings = get_settings()
    if not settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server belum dikonfigurasi. Hubungi administrator.",
        )
    if not api_key or api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key tidak valid.",
            headers={"WWW-Authenticate": "X-Internal-API-Key"},
        )
    return api_key


# Schemas

class RiasecInput(BaseModel):
    r: float = Field(..., ge=0, le=100)
    i: float = Field(..., ge=0, le=100)
    a: float = Field(..., ge=0, le=100)
    s: float = Field(..., ge=0, le=100)
    e: float = Field(..., ge=0, le=100)
    c: float = Field(..., ge=0, le=100)


class AkademikInput(BaseModel):
    logika: float = Field(..., ge=0, le=100)
    bahasa: float = Field(..., ge=0, le=100)
    sains: float = Field(..., ge=0, le=100)
    sosial: float = Field(..., ge=0, le=100)
    praktik: float = Field(..., ge=0, le=100)


class AnalyzeRequest(BaseModel):
    riasec: RiasecInput
    akademik: AkademikInput
    top_k: int = Field(default=3, ge=1, le=10)


# Custom Keras Layer

@tf.keras.utils.register_keras_serializable()
class RIASECAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attention_dense = tf.keras.layers.Dense(1, use_bias=False)
        self.projection = tf.keras.layers.Dense(units, activation="relu")

    def build(self, input_shape):
        self.attention_dense.build((input_shape[0], input_shape[1], 1))
        self.projection.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        x_expanded = tf.expand_dims(inputs, axis=-1)
        attn_scores = self.attention_dense(x_expanded)
        attn_weights = tf.nn.softmax(attn_scores, axis=1)
        weighted = inputs * tf.squeeze(attn_weights, axis=-1)
        return self.projection(weighted)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# Inference Service

MODEL_DIR      = os.getenv("MODEL_DIR", "model_artifacts")
RIASEC_ORDER   = ["r", "i", "a", "s", "e", "c"]
AKADEMIK_ORDER = ["logika", "bahasa", "sains", "sosial", "praktik"]

RIASEC_LABELS = {
    "R": "Realistik",
    "I": "Investigatif",
    "A": "Artistik",
    "S": "Sosial",
    "E": "Enterprising",
    "C": "Konvensional",
}


def _load_pickle(filename: str) -> Any:
    with open(os.path.join(MODEL_DIR, filename), "rb") as f:
        return pickle.load(f)


def _load_json(filename: str) -> Any:
    with open(os.path.join(MODEL_DIR, filename), "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    logger.info("Memuat model artifacts dari '%s' …", MODEL_DIR)
    arts = {
        "label_encoder":       _load_pickle("label_encoder.pkl"),
        "scaler_riasec":       _load_pickle("scaler_riasec.pkl"),
        "scaler_akademik":     _load_pickle("scaler_akademik.pkl"),
        "model_info":          _load_json("model_info.json"),
        "prodi_riasec_mean":   _load_json("prodi_riasec_mean.json"),
        "prodi_akademik_mean": _load_json("prodi_akademik_mean.json"),
        "rumpun_list":         _load_json("rumpun_list.json"),
        "rumpun_to_prodi":     _load_json("rumpun_to_prodi.json"),
    }
    arts["model"] = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "model_riasec.keras"),
        custom_objects={"RIASECAttentionLayer": RIASECAttentionLayer},
        compile=False,
        safe_mode=False,
    )
    logger.info("Model artifacts berhasil dimuat.")
    return arts


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def run_prediction(riasec: dict, akademik: dict, top_k: int = 3) -> dict:
    arts = load_artifacts()

    riasec_vec   = np.array([riasec[k]   for k in RIASEC_ORDER],   dtype=np.float32)
    akademik_vec = np.array([akademik[k] for k in AKADEMIK_ORDER], dtype=np.float32)

    riasec_scaled   = arts["scaler_riasec"].transform(riasec_vec.reshape(1, -1))
    akademik_scaled = arts["scaler_akademik"].transform(akademik_vec.reshape(1, -1))

    raw_pred = arts["model"].predict([riasec_scaled, akademik_scaled], verbose=0)

    pred_label  = arts["label_encoder"].inverse_transform([int(np.argmax(raw_pred, axis=1)[0])])[0]
    kode_riasec = "".join([k.upper() for k in RIASEC_ORDER][i] for i in np.argsort(riasec_vec)[::-1][:3])

    top3_idx = np.argsort(raw_pred[0])[::-1][:3]
    top_personality = [
        {
            "label":        arts["label_encoder"].inverse_transform([int(i)])[0],
            "probabilitas": round(float(raw_pred[0][i]) * 100, 2),
        }
        for i in top3_idx
    ]

    pred    = raw_pred[0]
    top_idx = np.argsort(pred)[::-1][:top_k]

    rekomendasi = []
    for idx in top_idx:
        rumpun    = arts["label_encoder"].classes_[idx]
        input_vec = np.concatenate([riasec_vec, akademik_vec])
        scores = []
        for prodi in arts["rumpun_to_prodi"].get(rumpun, []):
            if prodi not in arts["prodi_riasec_mean"] or prodi not in arts["prodi_akademik_mean"]:
                continue
            pv = np.array(
                list(arts["prodi_riasec_mean"][prodi].values()) +
                list(arts["prodi_akademik_mean"][prodi].values())
            )
            scores.append({"program_name": prodi, "raw": _cosine_similarity(input_vec, pv)})

        scores.sort(key=lambda x: x["raw"], reverse=True)
        total = sum(x["raw"] for x in scores[:3])
        prodi_final = (
            [{"program_name": p["program_name"], "similarity_persen": round(p["raw"] / total * 100, 1)}
             for p in scores[:3]]
            if total > 0 else []
        )
        rekomendasi.append({
            "rumpun":           rumpun,
            "kecocokan_persen": round(float(pred[idx]) * 100, 2),
            "prodi_tersedia":   prodi_final,
        })

    return {
        "kode_riasec":     kode_riasec,
        "prediksi_utama":  pred_label,
        "top_personality": top_personality,
        "rekomendasi":     rekomendasi,
    }


# Gemini Service

_gemini_client: Any = None

GEMINI_SYSTEM = (
    "Kamu adalah konselor pendidikan Indonesia yang profesional dan empatik. "
    "Tugas kamu adalah menjelaskan hasil analisis minat dan kemampuan "
    "kepada peserta secara langsung menggunakan kata 'kamu'. "
    "Jangan pernah menggunakan kata 'siswa ini', 'ia', atau 'mereka'. "
    "Balas HANYA dengan JSON valid, tanpa markdown, tanpa teks lain."
)


def _get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is None:
        settings = get_settings()
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY belum dikonfigurasi.")
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def _build_narasi_prompt(
    riasec: dict,
    akademik: dict,
    prediction: dict,
) -> str:
    """
    Bangun prompt kaya data agar Gemini menghasilkan narasi
    yang benar-benar personal — bukan template.
    """
    # Baris RIASEC dengan label
    riasec_lines = "\n".join(
        f"  {k.upper()} ({RIASEC_LABELS.get(k.upper(), k)}): {riasec[k]:.1f}"
        for k in RIASEC_ORDER
    )

    # Baris akademik
    akademik_lines = "\n".join(
        f"  {k.capitalize()}: {akademik[k]:.1f}"
        for k in AKADEMIK_ORDER
    )

    # Data prediksi
    top_personality = prediction.get("top_personality", [])
    utama = top_personality[0] if len(top_personality) > 0 else {}
    alt1  = top_personality[1] if len(top_personality) > 1 else {}
    alt2  = top_personality[2] if len(top_personality) > 2 else {}

    # RIASEC tertinggi & terendah
    riasec_sorted = sorted(
        [(k.upper(), riasec[k]) for k in RIASEC_ORDER],
        key=lambda x: x[1], reverse=True,
    )
    tertinggi = riasec_sorted[0]
    terendah  = riasec_sorted[-1]

    # Akademik tertinggi & terendah
    akademik_sorted = sorted(
        [(k.capitalize(), akademik[k]) for k in AKADEMIK_ORDER],
        key=lambda x: x[1], reverse=True,
    )
    ak_tertinggi = akademik_sorted[0]
    ak_terendah  = akademik_sorted[-1]

    utama_label = utama.get("label", "-")
    alt1_label  = alt1.get("label",  "-")

    # Siapkan string join di luar f-string (Python 3.11 tidak izinkan backslash dalam f-string)
    riasec_detail  = "; ".join(f"{k.upper()}={riasec[k]:.0f}" for k in RIASEC_ORDER)
    akademik_detail = "; ".join(f"{k.capitalize()}={akademik[k]:.0f}" for k in AKADEMIK_ORDER)
    utama_prob     = utama.get("probabilitas", 0)
    alt1_prob      = alt1.get("probabilitas", 0)
    alt2_label     = alt2.get("label", "-")
    alt2_prob      = alt2.get("probabilitas", 0)

    json_template = (
        '{"ringkasan":"2-3 kalimat langsung ke kamu tentang pola minat & kemampuanmu",'
        '"kekuatan":["konkret 1","konkret 2","konkret 3"],'
        f'"alasan_kecocokan":"mengapa {utama_label} cocok vs {alt1_label}, pakai angka",'
        '"saran_pengembangan":"1-2 kalimat saran konkret ke kamu"}'
    )

    prompt = (
        "## Peran & Gaya Bahasa\n"
        "Kamu adalah konselor pendidikan yang sedang berbicara langsung kepada peserta.\n"
        "Gunakan kata \"kamu\" — bukan \"siswa ini\", \"ia\", atau \"mereka\".\n\n"
        "## Data Tes\n"
        f"RIASEC: {tertinggi[0]}={tertinggi[1]:.0f}(tertinggi) {terendah[0]}={terendah[1]:.0f}(terendah) | "
        f"Detail: {riasec_detail}\n"
        f"Akademik: {ak_tertinggi[0]}={ak_tertinggi[1]:.0f}(terkuat) {ak_terendah[0]}={ak_terendah[1]:.0f}(terlemah) | "
        f"Detail: {akademik_detail}\n"
        f"Prediksi ML: {utama_label}={utama_prob:.1f}% | {alt1_label}={alt1_prob:.1f}% | {alt2_label}={alt2_prob:.1f}%\n\n"
        "## Instruksi\n"
        "Analisis personal berdasarkan angka. Jangan jelaskan teori RIASEC. "
        "Pakai perbandingan angka. Bahasa Indonesia hangat. "
        "WAJIB singkat: ringkasan maks 2 kalimat, kekuatan maks 3 item pendek (maks 8 kata per item), "
        "alasan_kecocokan maks 2 kalimat, saran_pengembangan maks 2 kalimat. "
        "Total output JSON tidak boleh melebihi 300 kata.\n\n"
        f"Balas HANYA JSON ini:\n{json_template}"
    )

    return prompt


def _call_gemini_narasi(
    riasec: dict,
    akademik: dict,
    prediction: dict,
    settings: Settings,
    max_retries: int = 2,
) -> dict:
    """Panggil Gemini dengan retry. Return dict terstruktur."""
    client = _get_gemini_client()
    prompt = _build_narasi_prompt(riasec, akademik, prediction)

    logger.info(
        "Gemini narasi | RIASEC: %s | model: %s | prompt: %d karakter",
        prediction.get("kode_riasec", "?"),
        settings.GEMINI_MODEL,
        len(prompt),
    )

    last_raw = ""
    for attempt in range(1, max_retries + 1):
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=GEMINI_SYSTEM,
                temperature=0.5,
                max_output_tokens=2048,
            ),
        )

        # Log finish_reason agar mudah debug
        try:
            finish_reason = response.candidates[0].finish_reason
            logger.info("Gemini finish_reason (attempt %d): %s", attempt, finish_reason)
            if str(finish_reason) == "MAX_TOKENS":
                logger.warning(
                    "Gemini terpotong karena MAX_TOKENS pada attempt %d. "
                    "Pertimbangkan naikkan max_output_tokens lebih lanjut.", attempt
                )
        except Exception:
            pass

        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        last_raw = raw

        logger.info("RAW GEMINI RESPONSE (attempt %d): %s", attempt, raw[:600])

        try:
            result = json.loads(raw)
            # Validasi minimal — semua field wajib ada
            required = {"ringkasan", "kekuatan", "alasan_kecocokan", "saran_pengembangan"}
            if required.issubset(result.keys()):
                return result
            logger.warning("Gemini response tidak lengkap (attempt %d): %s", attempt, list(result.keys()))
        except json.JSONDecodeError:
            logger.error("Gemini JSON parse gagal (attempt %d). Raw: %s", attempt, raw[:300])

    logger.error("Semua %d attempt Gemini gagal. Last raw: %s", max_retries, last_raw[:300])
    return {}


def _fallback_narasi(prediction: dict, riasec: dict, akademik: dict) -> dict:
    """Fallback terstruktur jika Gemini gagal — tetap informatif."""
    top = prediction.get("top_personality", [{}])
    utama_label = top[0].get("label", "bidang yang direkomendasikan") if top else "bidang yang direkomendasikan"

    riasec_sorted = sorted(
        [(k.upper(), riasec[k]) for k in RIASEC_ORDER],
        key=lambda x: x[1], reverse=True,
    )
    ak_sorted = sorted(
        [(k.capitalize(), akademik[k]) for k in AKADEMIK_ORDER],
        key=lambda x: x[1], reverse=True,
    )

    dom_r  = f"{riasec_sorted[0][0]} ({RIASEC_LABELS.get(riasec_sorted[0][0], '')})"
    dom_ak = ak_sorted[0][0]

    return {
        "ringkasan": (
            f"Kamu menunjukkan dominasi kuat pada dimensi {dom_r} "
            f"dengan kemampuan akademik terbaik di {dom_ak}. "
            f"Pola ini mendukung kecocokan kamu pada {utama_label}."
        ),
        "kekuatan": [
            f"Kemampuan {ak_sorted[0][0].lower()} yang menonjol (skor {ak_sorted[0][1]:.0f})",
            f"Minat kuat pada aktivitas {RIASEC_LABELS.get(riasec_sorted[0][0], '').lower()}",
            "Pola minat dan kemampuan yang konsisten dan terarah",
        ],
        "alasan_kecocokan": (
            f"Kombinasi minat dan kemampuan akademik kamu paling selaras dengan {utama_label}. "
            "Probabilitas dari model menunjukkan selisih yang signifikan dibanding alternatif lain."
        ),
        "saran_pengembangan": (
            f"Perkuat portofolio dan pengalaman praktis di bidang {utama_label} "
            "untuk meningkatkan daya saing kamu saat seleksi program studi."
        ),
    }


# Unified Stream Generator

async def analyze_and_stream(
    riasec: dict,
    akademik: dict,
    top_k: int,
) -> AsyncGenerator[str, None]:

    # Step 1: prediksi model — kirim langsung tanpa tunggu Gemini
    prediction = run_prediction(riasec, akademik, top_k)
    yield f"event: prediction\ndata: {json.dumps(prediction, ensure_ascii=False)}\n\n"

    # Step 2: narasi terstruktur dari Gemini
    settings = get_settings()

    try:
        narasi_dict = await asyncio.to_thread(
            _call_gemini_narasi, riasec, akademik, prediction, settings
        )
        if not narasi_dict:
            narasi_dict = _fallback_narasi(prediction, riasec, akademik)
    except Exception as e:
        logger.exception("Gemini narasi error: %s", e)
        narasi_dict = _fallback_narasi(prediction, riasec, akademik)

    payload = {
        "kode_riasec": prediction["kode_riasec"],
        **narasi_dict,  # ringkasan, kekuatan, alasan_kecocokan, saran_pengembangan
    }
    yield f"event: narasi\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


# App & Middleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API berhasil dijalankan")
    yield
    logger.info("API dimatikan")


app = FastAPI(
    title="Capstone Recommendation API",
    version="4.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Endpoints

@app.api_route("/health", methods=["GET", "HEAD"], tags=["System"])
async def health():
    return JSONResponse(content={"status": "ok", "service": "capstone-fastapi", "version": "4.1.0"})


@app.get("/", tags=["System"])
async def root():
    return JSONResponse(content={
        "message": "KompasKarir AI",
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "analyze": "POST /api/analyze"
        }
    })


@app.post(
    "/api/analyze",
    tags=["Analyze"],
    summary="Prediksi RIASEC + narasi kepribadian dalam satu SSE stream",
    response_description=(
        "SSE stream:\n"
        "  event: prediction → hasil prediksi model TensorFlow\n"
        "  event: narasi     → narasi terstruktur dari Gemini (ringkasan, kekuatan, alasan_kecocokan, saran_pengembangan)\n"
        "  data: [DONE]      → selesai"
    ),
)
async def analyze(
    body: AnalyzeRequest,
    _: Annotated[str, Depends(verify_internal_key)],
) -> StreamingResponse:
    try:
        generator = analyze_and_stream(
            riasec=body.riasec.model_dump(),
            akademik=body.akademik.model_dump(),
            top_k=body.top_k,
        )
    except Exception as e:
        logger.error("Error inisialisasi stream: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )