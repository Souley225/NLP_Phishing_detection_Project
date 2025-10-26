"""
Évaluation du modèle entraîné sur le test set.

Ce script charge le modèle entraîné et les feature extractors,
puis évalue les performances sur le test set avec visualisations.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import load_config, validate_config
from src.features.build_features import URLFeatureExtractor
from src.utils.io_utils import ensure_dir, load_csv, load_joblib
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.metrics import (
    calculate_metrics,
    get_classification_report,
    get_confusion_matrix,
    print_metrics,
)

# Configuration du logger
setup_logging(level="INFO", log_format="text")
logger = get_logger(__name__)


def load_model_and_extractors(cfg: any) -> tuple:
    """
    Charge le modèle et les feature extractors.
    
    Args:
        cfg: Configuration Hydra
    
    Returns:
        Tuple (model, feature_extractor)
    """
    logger.info("Chargement du modèle et des extractors...")
    
    models_dir = Path(cfg.paths.models_dir)
    
    # Charger le modèle
    model_path = models_dir / cfg.model.default.save_name
    model = load_joblib(model_path)
    logger.info(f"✓ Modèle chargé: {model_path}")
    
    # Reconstruire le feature extractor
    feature_extractor = URLFeatureExtractor(cfg.features)
    
    # Charger les vectorizers
    if cfg.features.tfidf_word.use:
        tfidf_word_path = models_dir / cfg.model.default.vectorizers.tfidf_word
        feature_extractor.tfidf_word = load_joblib(tfidf_word_path)
        logger.info(f"✓ TF-IDF mots chargé: {tfidf_word_path}")
    
    if cfg.features.tfidf_char.use:
        tfidf_char_path = models_dir / cfg.model.default.vectorizers.tfidf_char
        feature_extractor.tfidf_char = load_joblib(tfidf_char_path)
        logger.info(f"✓ TF-IDF char chargé: {tfidf_char_path}")
    
    if cfg.features.lexical.use:
        scaler_path = models_dir / cfg.model.default.vectorizers.scaler
        feature_extractor.scaler = load_joblib(scaler_path)
        logger.info(f"✓ Scaler chargé: {scaler_path}")
    
    feature_extractor.is_fitted = True
    
    return model, feature_extractor


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """
    Crée et sauvegarde la matrice de confusion.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        save_path: Chemin de sauvegarde
    """
    logger.info("Génération de la matrice de confusion...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        cmap="Blues",
        display_labels=["Légitime", "Phishing"],
    )
    
    ax.set_title("Matrice de Confusion", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"✓ Matrice de confusion sauvegardée: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path) -> None:
    """
    Crée et sauvegarde la courbe ROC.
    
    Args:
        y_true: Vraies étiquettes
        y_proba: Probabilités prédites
        save_path: Chemin de sauvegarde
    """
    logger.info("Génération de la courbe ROC...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Utiliser les probabilités de la classe positive (1 = phishing)
    if y_proba.ndim == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba
    
    RocCurveDisplay.from_predictions(
        y_true,
        y_proba_pos,
        ax=ax,
        name="Phishing Detector",
    )
    
    ax.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    ax.set_title("Courbe ROC", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"✓ Courbe ROC sauvegardée: {save_path}")


def main() -> None:
    """
    Fonction principale d'évaluation.
    """
    logger.info("=" * 80)
    logger.info("ÉVALUATION DU MODÈLE SUR TEST SET")
    logger.info("=" * 80)
    
    # Charger la configuration
    cfg = load_config()
    validate_config(cfg)
    
    # Charger le modèle et les extractors
    model, feature_extractor = load_model_and_extractors(cfg)
    
    # Charger le test set
    logger.info("Chargement du test set...")
    test_path = Path(cfg.paths.processed_dir) / "test.csv"
    test_df = load_csv(test_path)
    logger.info(f"Test set: {len(test_df)} samples")
    
    # Extraire X et y
    text_col = cfg.data.text_column
    target_col = cfg.data.target_column
    
    X_test_urls = test_df[text_col]
    y_test = test_df[target_col].values
    
    # Transformer les features
    logger.info("Transformation des features...")
    X_test = feature_extractor.transform(X_test_urls)
    logger.info(f"Shape X_test: {X_test.shape}")
    
    # Prédictions
    logger.info("Génération des prédictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculer les métriques
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics, title="Métriques Test Set")
    
    # Rapport de classification
    logger.info("\n" + "=" * 80)
    logger.info("RAPPORT DE CLASSIFICATION")
    logger.info("=" * 80)
    report = get_classification_report(
        y_test,
        y_pred,
        target_names=["Légitime", "Phishing"],
    )
    print(report)
    
    # Créer le répertoire pour les figures
    models_dir = Path(cfg.paths.models_dir)
    ensure_dir(models_dir)
    
    # Générer les visualisations
    cm_path = models_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, cm_path)
    
    roc_path = models_dir / "roc_curve.png"
    plot_roc_curve(y_test, y_proba, roc_path)
    
    logger.info("=" * 80)
    logger.info("✓ ÉVALUATION TERMINÉE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()