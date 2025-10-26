"""
Tests pour le module d'entraînement.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.utils.metrics import (
    calculate_metrics,
    get_classification_report,
    get_confusion_matrix,
)


def test_calculate_metrics():
    """Test du calcul des métriques."""
    # Prédictions parfaites
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.95, 0.05],
        [0.1, 0.9],
        [0.85, 0.15],
        [0.15, 0.85],
    ])
    
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # Vérifications
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    
    # Prédictions parfaites = métriques à 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_calculate_metrics_without_proba():
    """Test du calcul des métriques sans probabilités."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    
    metrics = calculate_metrics(y_true, y_pred, y_proba=None)
    
    # ROC-AUC ne doit pas être présent
    assert "roc_auc" not in metrics
    assert "accuracy" in metrics


def test_get_confusion_matrix():
    """Test de la matrice de confusion."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    
    cm = get_confusion_matrix(y_true, y_pred)
    
    # Vérifier la forme
    assert cm.shape == (2, 2)
    
    # Vérifier les valeurs (TP, TN, FP, FN)
    assert cm[0, 0] == 3  # True Negatives
    assert cm[1, 1] == 2  # True Positives
    assert cm[0, 1] == 0  # False Positives
    assert cm[1, 0] == 1  # False Negatives


def test_get_classification_report():
    """Test du rapport de classification."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    
    report = get_classification_report(
        y_true,
        y_pred,
        target_names=["Légitime", "Phishing"],
    )
    
    # Vérifier que le rapport est une chaîne
    assert isinstance(report, str)
    assert "Légitime" in report
    assert "Phishing" in report
    assert "precision" in report
    assert "recall" in report


def test_logistic_regression_training():
    """Test d'entraînement simple avec LogisticRegression."""
    # Données de test simples
    X_train = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Entraîner le modèle
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Vérifier que le modèle est entraîné
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    
    # Faire des prédictions
    y_pred = model.predict(X_train)
    
    # Vérifier que les prédictions sont valides
    assert len(y_pred) == len(y_train)
    assert all(p in [0, 1] for p in y_pred)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])