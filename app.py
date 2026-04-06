"""
FastAPI server for the Hugging Face SiEBERT sentiment model.

Endpoints:
    POST /predict     — Analyze text sentiment
    GET  /model-info  — Model architecture & training metrics
    GET  /health      — Health check

Run:
    cd d:\\Projects\\DL\\backend
    uvicorn app:app --reload --port 8000
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from predict import SentimentPredictor

load_dotenv()

# ─── App ─────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="SiEBERT (RoBERTa) Sentiment Model for Sentiment Classification",
    version="1.0.0",
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Predictor (loaded once at startup) ─────────────────────
predictor = SentimentPredictor()


@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts."""
    success = predictor.load()
    if not success:
        print("WARNING: Sentiment backend failed to load.")
        print("         The /predict endpoint will return 503 until backend is available.")


# ─── Schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to analyze for sentiment",
        json_schema_extra={"examples": ["This movie was absolutely fantastic! I loved every moment of it."]},
    )


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    input_length: int
    cleaned_text_preview: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ─── Endpoints ───────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — reports model status."""
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Analyze text sentiment.
    Returns label (Positive/Negative), confidence score, and per-class probabilities.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model backend not loaded. Verify backend configuration, network access, and HF_API_TOKEN (if using hosted inference).",
        )

    try:
        result = predictor.predict(request.text)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Return model architecture, training metrics, and configuration."""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model backend not loaded. Verify backend configuration, network access, and HF_API_TOKEN (if using hosted inference).",
        )
    return predictor.get_model_info()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
