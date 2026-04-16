"""
API FastAPI pour la détection de phishing.

Endpoints:
- GET /health: Health check
- POST /predict: Prédiction sur une URL

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is on path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_features import URLFeatureExtractor
from src.utils.io_utils import load_joblib

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Phishing Detection API",
    description="API de détection de phishing par analyse d'URLs",
    version="1.0.0",
    contact={"name": "Souleymane Sall", "email": "sallsouleymane2207@gmail.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Any = None
feature_extractor: URLFeatureExtractor | None = None


# ── Pydantic models ───────────────────────────────────────────────────────────

class URLRequest(BaseModel):
    url: str = Field(..., description="URL to analyse")


class PredictionResponse(BaseModel):
    url: str
    prediction: int
    label: str
    confidence: float
    proba_legitimate: float
    proba_phishing: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


# ── Startup ───────────────────────────────────────────────────────────────────

def _load_feature_config() -> dict:
    """Load the features section from configs/config.yaml using plain YAML."""
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["features"]


def _models_dir() -> Path:
    """Resolve the models directory relative to project root."""
    return ROOT / "models"


def load_model_artifacts() -> None:
    global model, feature_extractor

    logger.info("Loading model artifacts...")
    models_dir = _models_dir()

    # Load trained classifier
    model_path = models_dir / "best_model.pkl"
    model = load_joblib(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Load feature config and reconstruct extractor
    feat_cfg = _load_feature_config()
    feature_extractor = URLFeatureExtractor(feat_cfg)

    if feat_cfg.get("tfidf_word", {}).get("use"):
        feature_extractor.tfidf_word = load_joblib(models_dir / "tfidf_word.pkl")
        logger.info("tfidf_word loaded")

    if feat_cfg.get("tfidf_char", {}).get("use"):
        feature_extractor.tfidf_char = load_joblib(models_dir / "tfidf_char.pkl")
        logger.info("tfidf_char loaded")

    if feat_cfg.get("lexical", {}).get("use"):
        feature_extractor.scaler = load_joblib(models_dir / "scaler.pkl")
        logger.info("scaler loaded")

    feature_extractor.is_fitted = True
    logger.info("All artifacts loaded successfully.")


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=" * 60)
    logger.info("PHISHING DETECTION API — STARTUP")
    logger.info("=" * 60)
    try:
        load_model_artifacts()
        logger.info("API ready.")
    except Exception as e:
        # Log but do NOT raise — FastAPI starts anyway; /health reports unhealthy
        logger.error(f"Model loading failed: {e}", exc_info=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root() -> dict:
    return {
        "name": "Phishing Detection API",
        "version": "1.0.0",
        "endpoints": {"/health": "GET", "/predict": "POST", "/docs": "GET"},
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: URLRequest) -> PredictionResponse:
    if model is None or feature_extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service initialising.")

    try:
        logger.info(f"Predicting: {request.url}")
        X = feature_extractor.transform(pd.Series([request.url]))

        raw_pred    = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Handle both int (0/1) and string ("good"/"bad") labels
        LABEL_TO_INT = {"bad": 1, "good": 0}
        prediction = (
            int(raw_pred)
            if isinstance(raw_pred, (int, np.integer))
            else LABEL_TO_INT.get(str(raw_pred), 0)
        )

        # Map class order → proba indices
        classes = list(model.classes_)
        if isinstance(classes[0], (int, np.integer)):
            idx_legit = classes.index(0) if 0 in classes else 0
            idx_phish = classes.index(1) if 1 in classes else 1
        else:
            idx_legit = classes.index("good") if "good" in classes else 0
            idx_phish = classes.index("bad")  if "bad"  in classes else 1

        proba_legitimate = float(probabilities[idx_legit])
        proba_phishing   = float(probabilities[idx_phish])
        confidence       = proba_phishing if prediction == 1 else proba_legitimate

        result = PredictionResponse(
            url=request.url,
            prediction=prediction,
            label="phishing" if prediction == 1 else "legitimate",
            confidence=confidence,
            proba_legitimate=proba_legitimate,
            proba_phishing=proba_phishing,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        logger.info(f"Result: {result.label} ({result.confidence:.2%})")
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
