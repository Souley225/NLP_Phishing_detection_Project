"""
API FastAPI pour la détection de phishing.

Cette API expose des endpoints REST pour prédire si une URL
est légitime ou du phishing.

Endpoints:
- GET /health: Health check
- POST /predict: Prédiction sur une URL

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import load_config, validate_config
from src.features.build_features import URLFeatureExtractor
from src.utils.io_utils import load_joblib
from src.utils.logging_utils import get_logger, setup_logging

# Configuration du logger
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"), log_format="json")
logger = get_logger(__name__)

# Initialiser FastAPI
app = FastAPI(
    title="Phishing Detection API",
    description="API de détection de phishing par analyse d'URLs",
    version="1.0.0",
    contact={
        "name": "Souleymane Sall",
        "email": "sallsouleymane2207@gmail.com",
    },
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour le modèle et les extractors
model: Any = None
feature_extractor: URLFeatureExtractor | None = None
config: Any = None


# Modèles Pydantic pour validation
class URLRequest(BaseModel):
    """Requête de prédiction pour une URL."""
    
    url: str = Field(
        ...,
        description="URL à analyser",
        example="http://paypal-secure.tk/login.php",
    )


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    
    url: str = Field(..., description="URL analysée")
    prediction: int = Field(..., description="0=légitime, 1=phishing")
    label: str = Field(..., description="Label textuel de la prédiction")
    confidence: float = Field(..., description="Confiance de la prédiction (0-1)")
    proba_legitimate: float = Field(..., description="Probabilité légitime")
    proba_phishing: float = Field(..., description="Probabilité phishing")
    timestamp: str = Field(..., description="Timestamp de la prédiction")


class HealthResponse(BaseModel):
    """Réponse du health check."""
    
    status: str = Field(..., description="Statut du service")
    model_loaded: bool = Field(..., description="Modèle chargé ou non")
    timestamp: str = Field(..., description="Timestamp du check")


def load_model_artifacts() -> None:
    """
    Charge le modèle et les feature extractors au démarrage de l'API.
    
    Raises:
        Exception: Si le chargement échoue
    """
    global model, feature_extractor, config
    
    logger.info("Chargement des artifacts du modèle...")
    
    try:
        # Charger la configuration
        config = load_config()
        validate_config(config)
        
        models_dir = Path(config.paths.models_dir)
        
        # Charger le modèle
        model_path = models_dir / config.model.default.save_name
        model = load_joblib(model_path)
        logger.info(f"✓ Modèle chargé: {model_path}")
        
        # Reconstruire le feature extractor
        feature_extractor = URLFeatureExtractor(config.features)
        
        # Charger les vectorizers
        if config.features.tfidf_word.use:
            tfidf_word_path = models_dir / config.model.default.vectorizers.tfidf_word
            feature_extractor.tfidf_word = load_joblib(tfidf_word_path)
            logger.info(f"✓ TF-IDF mots chargé")
        
        if config.features.tfidf_char.use:
            tfidf_char_path = models_dir / config.model.default.vectorizers.tfidf_char
            feature_extractor.tfidf_char = load_joblib(tfidf_char_path)
            logger.info(f"✓ TF-IDF char chargé")
        
        if config.features.lexical.use:
            scaler_path = models_dir / config.model.default.vectorizers.scaler
            feature_extractor.scaler = load_joblib(scaler_path)
            logger.info(f"✓ Scaler chargé")
        
        feature_extractor.is_fitted = True
        
        logger.info("✓ Tous les artifacts chargés avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des artifacts: {e}")
        raise


@app.on_event("startup")
async def startup_event() -> None:
    """
    Événement de démarrage de l'API.
    
    Charge le modèle et les artifacts au démarrage.
    """
    logger.info("=" * 80)
    logger.info("DÉMARRAGE DE L'API PHISHING DETECTION")
    logger.info("=" * 80)
    
    try:
        load_model_artifacts()
        logger.info("✓ API prête à servir les requêtes")
    except Exception as e:
        logger.error(f"Échec du démarrage: {e}")
        # En production, on pourrait vouloir arrêter l'app ici
        raise


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """
    Endpoint racine avec informations de base.
    
    Returns:
        Informations sur l'API
    """
    return {
        "name": "Phishing Detection API",
        "version": "1.0.0",
        "author": "Souleymane Sall",
        "email": "sallsouleymane2207@gmail.com",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Vérifie que l'API fonctionne et que le modèle est chargé.
    
    Returns:
        Statut de santé de l'API
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: URLRequest) -> PredictionResponse:
    """
    Prédit si une URL est du phishing ou légitime.
    
    Args:
        request: Requête contenant l'URL à analyser
    
    Returns:
        Prédiction avec probabilités
    
    Raises:
        HTTPException: Si le modèle n'est pas chargé ou si la prédiction échoue
    """
    # Vérifier que le modèle est chargé
    if model is None or feature_extractor is None:
        logger.error("Modèle non chargé")
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Service en cours d'initialisation.",
        )
    
    try:
        # Logger la requête
        logger.info(f"Prédiction demandée pour URL: {request.url}")
        
        # Transformer l'URL en features
        X = feature_extractor.transform(pd.Series([request.url]))
        
        # Prédiction
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        
        # Préparer la réponse
        response = PredictionResponse(
            url=request.url,
            prediction=prediction,
            label="phishing" if prediction == 1 else "legitimate",
            confidence=float(probabilities[prediction]),
            proba_legitimate=float(probabilities[0]),
            proba_phishing=float(probabilities[1]),
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        
        # Logger le résultat
        logger.info(f"Prédiction: {response.label} (confiance: {response.confidence:.2%})")
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    
    # Récupérer le port depuis les variables d'environnement
    port = int(os.getenv("PORT", "8000"))
    
    # Lancer le serveur
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )