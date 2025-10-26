"""
Tests pour le module de feature engineering.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    URLFeatureExtractor,
    calculate_entropy,
    extract_lexical_features,
    tokenize_url,
)


def test_tokenize_url():
    """Test de la tokenisation d'URL."""
    # URL simple
    url = "http://example.com/path"
    tokens = tokenize_url(url)
    assert "http" in tokens
    assert "example" in tokens
    assert "com" in tokens
    assert "path" in tokens
    
    # URL complexe avec paramètres
    url = "https://login.example.com/user-account.php?id=123&action=verify"
    tokens = tokenize_url(url)
    assert "login" in tokens
    assert "user" in tokens
    assert "account" in tokens


def test_calculate_entropy():
    """Test du calcul d'entropie."""
    # Chaîne uniforme (faible entropie)
    text1 = "aaaa"
    entropy1 = calculate_entropy(text1)
    assert entropy1 == 0.0  # Tous les caractères identiques
    
    # Chaîne variée (haute entropie)
    text2 = "abcdefgh"
    entropy2 = calculate_entropy(text2)
    assert entropy2 > 0.0
    
    # Chaîne vide
    text3 = ""
    entropy3 = calculate_entropy(text3)
    assert entropy3 == 0.0


def test_extract_lexical_features():
    """Test de l'extraction des features lexicales."""
    # URL légitime
    url = "https://www.google.com/search"
    features = extract_lexical_features(url)
    
    # Vérifier que toutes les features sont présentes
    expected_keys = [
        "url_length", "num_dots", "num_slashes", "num_dashes",
        "num_underscores", "num_at", "num_question", "num_ampersand",
        "num_equal", "num_digits", "digit_ratio", "has_ip",
        "entropy", "subdomain_count", "path_depth", "suspicious_tld"
    ]
    
    for key in expected_keys:
        assert key in features
        assert isinstance(features[key], (int, float))
    
    # Vérifications spécifiques
    assert features["url_length"] == len(url)
    assert features["num_dots"] == 2
    assert features["num_slashes"] == 3
    assert features["has_ip"] == 0.0  # Pas d'IP
    assert features["suspicious_tld"] == 0.0  # TLD légitime


def test_extract_lexical_features_with_ip():
    """Test avec une URL contenant une IP."""
    url = "http://192.168.1.1/login"
    features = extract_lexical_features(url)
    
    assert features["has_ip"] == 1.0
    assert features["num_dots"] == 3  # Dans l'IP


def test_extract_lexical_features_suspicious_tld():
    """Test avec un TLD suspect."""
    url = "http://phishing-site.tk/verify"
    features = extract_lexical_features(url)
    
    assert features["suspicious_tld"] == 1.0


def test_url_feature_extractor_initialization():
    """Test de l'initialisation du feature extractor."""
    config = {
        "tfidf_word": {
            "use": True,
            "ngram_range": [1, 2],
            "max_features": 100,
            "min_df": 1,
            "max_df": 1.0,
        },
        "tfidf_char": {
            "use": True,
            "analyzer": "char",
            "ngram_range": [2, 4],
            "max_features": 100,
            "min_df": 1,
            "max_df": 1.0,
        },
        "lexical": {
            "use": True,
            "features": [
                "url_length", "num_dots", "entropy",
            ],
        },
    }
    
    extractor = URLFeatureExtractor(config)
    
    assert extractor.tfidf_word is not None
    assert extractor.tfidf_char is not None
    assert extractor.scaler is not None
    assert not extractor.is_fitted


def test_url_feature_extractor_fit_transform():
    """Test du fit et transform du feature extractor."""
    # Configuration minimale
    config = {
        "tfidf_word": {
            "use": True,
            "ngram_range": [1, 2],
            "max_features": 50,
            "min_df": 1,
            "max_df": 1.0,
        },
        "tfidf_char": {
            "use": False,
            "analyzer": "char",
            "ngram_range": [2, 4],
            "max_features": 50,
            "min_df": 1,
            "max_df": 1.0,
        },
        "lexical": {
            "use": True,
            "features": [
                "url_length", "num_dots", "entropy",
            ],
        },
    }
    
    # Données de test
    urls = pd.Series([
        "http://google.com",
        "http://example.com/page",
        "http://phishing.tk/login",
    ])
    
    extractor = URLFeatureExtractor(config)
    
    # Fit et transform
    X = extractor.fit_transform(urls)
    
    assert extractor.is_fitted
    assert X.shape[0] == len(urls)
    assert X.shape[1] > 0  # Au moins quelques features
    assert isinstance(X, np.ndarray)


def test_url_feature_extractor_transform_without_fit():
    """Test du transform sans fit (doit lever une erreur)."""
    config = {
        "tfidf_word": {"use": True, "ngram_range": [1, 2], "max_features": 50, "min_df": 1, "max_df": 1.0},
        "tfidf_char": {"use": False, "analyzer": "char", "ngram_range": [2, 4], "max_features": 50, "min_df": 1, "max_df": 1.0},
        "lexical": {"use": True, "features": ["url_length"]},
    }
    
    urls = pd.Series(["http://example.com"])
    extractor = URLFeatureExtractor(config)
    
    with pytest.raises(ValueError, match="doit être entraîné"):
        extractor.transform(urls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])