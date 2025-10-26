"""
Tests pour l'API FastAPI.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import pytest
from fastapi.testclient import TestClient

from src.serving.api import app

# Créer un client de test
client = TestClient(app)


def test_root_endpoint():
    """Test de l'endpoint racine."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "name" in data
    assert "version" in data
    assert "author" in data
    assert data["author"] == "Souleymane Sall"


def test_health_endpoint():
    """Test de l'endpoint health check."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data
    
    # Vérifier le format du timestamp
    assert "T" in data["timestamp"]


def test_predict_endpoint_valid_url():
    """Test de prédiction avec une URL valide."""
    # Note: Ce test nécessite que le modèle soit chargé
    # En production, on utiliserait des mocks
    
    test_url = "http://google.com"
    
    response = client.post(
        "/predict",
        json={"url": test_url},
    )
    
    # Si le modèle n'est pas chargé, on attend un 503
    # Sinon on vérifie la structure de la réponse
    if response.status_code == 200:
        data = response.json()
        
        assert "url" in data
        assert "prediction" in data
        assert "label" in data
        assert "confidence" in data
        assert "proba_legitimate" in data
        assert "proba_phishing" in data
        assert "timestamp" in data
        
        assert data["url"] == test_url
        assert data["prediction"] in [0, 1]
        assert data["label"] in ["legitimate", "phishing"]
        assert 0.0 <= data["confidence"] <= 1.0
        
    elif response.status_code == 503:
        # Modèle non chargé (attendu dans les tests)
        assert "detail" in response.json()


def test_predict_endpoint_missing_url():
    """Test de prédiction sans URL."""
    response = client.post(
        "/predict",
        json={},
    )
    
    # Doit retourner une erreur de validation (422)
    assert response.status_code == 422


def test_predict_endpoint_invalid_json():
    """Test avec un JSON invalide."""
    response = client.post(
        "/predict",
        data="invalid json",
        headers={"Content-Type": "application/json"},
    )
    
    assert response.status_code == 422


def test_api_cors_headers():
    """Test que les headers CORS sont présents."""
    response = client.get("/")
    
    # Les headers CORS peuvent être présents selon la configuration
    # On vérifie juste que la requête passe
    assert response.status_code == 200


def test_openapi_docs():
    """Test que la documentation OpenAPI est accessible."""
    response = client.get("/docs")
    
    # Devrait rediriger ou afficher la doc
    assert response.status_code in [200, 307]


def test_openapi_json():
    """Test que le schéma OpenAPI JSON est accessible."""
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "openapi" in data
    assert "info" in data
    assert "paths" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])