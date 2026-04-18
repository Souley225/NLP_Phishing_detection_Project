"""
Script d'entraînement pour le déploiement.

Entraîne le modèle avec les hyperparamètres du config, calcule le seuil
optimal sur la validation et sauvegarde tous les artefacts dans models/.
Pas de MLflow, pas d'Optuna — conçu pour tourner en dehors de Hydra.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import URLFeatureExtractor, normalize_url
from src.utils.io_utils import ensure_dir, load_csv, save_joblib
from src.utils.metrics import find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
LABEL_MAP = {"bad": 1, "good": 0}


def _load_config() -> dict:
    """Fusionne config.yaml, model/default.yaml et train/default.yaml sans Hydra."""
    def _read(p: Path) -> dict:
        with open(p) as f:
            return yaml.safe_load(f)

    cfg = _read(ROOT / "configs" / "config.yaml")
    cfg["model"] = _read(ROOT / "configs" / "model" / "default.yaml")
    cfg["train"] = _read(ROOT / "configs" / "train" / "default.yaml")
    return cfg


def main() -> None:
    logger.info("=" * 60)
    logger.info("ENTRAINEMENT DU MODELE DE DETECTION DE PHISHING")
    logger.info("=" * 60)

    cfg = _load_config()

    text_col   = cfg["data"]["text_column"]
    target_col = cfg["data"]["target_column"]
    processed_dir = Path(cfg["paths"]["processed_dir"])

    # Chargement des donnees
    train_df = load_csv(processed_dir / "train.csv")
    val_df   = load_csv(processed_dir / "val.csv")
    logger.info("Train: %d | Val: %d", len(train_df), len(val_df))

    # Encodage des labels textuels en entiers
    train_df[target_col] = train_df[target_col].map(LABEL_MAP)
    val_df[target_col]   = val_df[target_col].map(LABEL_MAP)

    X_train_urls = train_df[text_col].map(normalize_url)
    y_train      = train_df[target_col].values
    X_val_urls   = val_df[text_col].map(normalize_url)
    y_val        = val_df[target_col].values

    # Extraction des features
    logger.info("Extraction des features...")
    feat_cfg = cfg["features"]
    feature_extractor = URLFeatureExtractor(feat_cfg)
    X_train = feature_extractor.fit_transform(X_train_urls)
    X_val   = feature_extractor.transform(X_val_urls)
    logger.info("X_train shape: %s", X_train.shape)

    # Entrainement
    model_params = dict(cfg["model"]["default"]["params"])
    model_params["random_state"] = cfg["train"]["default"]["seed"]
    logger.info("Parametres LinearSVC: %s", model_params)
    # CalibratedClassifierCV donne predict_proba à LinearSVC (isotonic sur la val)
    svc   = LinearSVC(**model_params)
    model = CalibratedClassifierCV(svc, cv=3, method="isotonic")
    model.fit(X_train, y_train)

    # Evaluation au seuil 0.5 (reference)
    y_pred = model.predict(X_val)
    logger.info("F1 (seuil=0.5): %.4f", f1_score(y_val, y_pred))
    logger.info(
        "\n%s",
        classification_report(y_val, y_pred, target_names=["legitime", "phishing"]),
    )

    # Recherche du seuil optimal sur la validation
    logger.info("Recherche du seuil optimal sur la validation...")
    y_proba = model.predict_proba(X_val)
    thr = find_optimal_threshold(y_val, y_proba)
    logger.info(
        "Seuil optimal: %.2f — F1=%.4f  P=%.4f  R=%.4f",
        thr["threshold"], thr["f1"], thr["precision"], thr["recall"],
    )

    # Sauvegarde des artefacts
    models_dir = Path(cfg["paths"]["models_dir"])
    ensure_dir(models_dir)

    model_cfg = cfg["model"]["default"]
    save_joblib(model, models_dir / model_cfg["save_name"])
    logger.info("Modele sauvegarde: %s", models_dir / model_cfg["save_name"])

    save_joblib(feature_extractor.tfidf_word, models_dir / model_cfg["vectorizers"]["tfidf_word"])
    save_joblib(feature_extractor.tfidf_char, models_dir / model_cfg["vectorizers"]["tfidf_char"])
    save_joblib(feature_extractor.scaler,     models_dir / model_cfg["vectorizers"]["scaler"])
    logger.info("Vectorizers sauvegardes.")

    threshold_path = models_dir / "threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(thr, f, indent=2)
    logger.info("Seuil sauvegarde: %s", threshold_path)

    with open(models_dir / "label_map.json", "w") as f:
        json.dump({"bad": 1, "good": 0}, f)
    logger.info("Label map sauvegarde.")

    logger.info("Entrainement termine. Artefacts dans: %s", models_dir)


if __name__ == "__main__":
    main()
