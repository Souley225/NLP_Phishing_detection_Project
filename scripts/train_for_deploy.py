"""
Quick training script for deployment (no MLflow, no Optuna).

Trains the phishing detection model with default hyperparameters
and saves all artifacts to models/.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config, validate_config
from src.features.build_features import URLFeatureExtractor
from src.utils.io_utils import ensure_dir, load_csv, save_joblib

LABEL_MAP = {"bad": 1, "good": 0}


def main() -> None:
    print("=" * 60)
    print("TRAINING PHISHING DETECTION MODEL (DEPLOY BUILD)")
    print("=" * 60)

    cfg = load_config()
    validate_config(cfg)

    text_col = cfg.data.text_column
    target_col = cfg.data.target_column

    # Load data
    train_df = load_csv(Path(cfg.paths.processed_dir) / "train.csv")
    val_df = load_csv(Path(cfg.paths.processed_dir) / "val.csv")
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Encode labels
    train_df[target_col] = train_df[target_col].map(LABEL_MAP)
    val_df[target_col] = val_df[target_col].map(LABEL_MAP)

    X_train_urls = train_df[text_col]
    y_train = train_df[target_col].values
    X_val_urls = val_df[text_col]
    y_val = val_df[target_col].values

    # Feature engineering
    print("Extracting features...")
    feature_extractor = URLFeatureExtractor(cfg.features)
    X_train = feature_extractor.fit_transform(X_train_urls)
    X_val = feature_extractor.transform(X_val_urls)
    print(f"X_train shape: {X_train.shape}")

    # Train with default hyperparameters from config
    params = dict(cfg.model.default.params)
    params["random_state"] = cfg.train.default.seed
    print(f"Training LogisticRegression with params: {params}")
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print(f"\nValidation F1: {f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=["legitimate", "phishing"]))

    # Save artifacts
    models_dir = Path(cfg.paths.models_dir)
    ensure_dir(models_dir)

    save_joblib(model, models_dir / cfg.model.default.save_name)
    print(f"Model saved: {models_dir / cfg.model.default.save_name}")

    save_joblib(feature_extractor.tfidf_word, models_dir / cfg.model.default.vectorizers.tfidf_word)
    save_joblib(feature_extractor.tfidf_char, models_dir / cfg.model.default.vectorizers.tfidf_char)
    save_joblib(feature_extractor.scaler, models_dir / cfg.model.default.vectorizers.scaler)
    print("Vectorizers saved.")

    # Save label map for API use
    import json
    with open(models_dir / "label_map.json", "w") as f:
        json.dump({"bad": 1, "good": 0}, f)
    print("Label map saved.")

    print("\n✓ Training complete. Artifacts saved to:", models_dir)


if __name__ == "__main__":
    main()
