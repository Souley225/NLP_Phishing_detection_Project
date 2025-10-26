"""
Pipeline d'entraînement avec Hydra + Optuna + MLflow.

Ce script implémente l'entraînement du modèle de détection de phishing
avec optimisation des hyperparamètres et tracking des expériences.

Workflow:
1. Chargement des données train/val
2. Feature engineering (TF-IDF + lexical)
3. Optimisation hyperparamètres avec Optuna
4. Entraînement du meilleur modèle
5. Logging dans MLflow
6. Sauvegarde des artifacts

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import random
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import load_config, print_config, save_config, validate_config
from src.features.build_features import URLFeatureExtractor
from src.utils.io_utils import ensure_dir, load_csv, save_joblib
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.metrics import calculate_metrics, print_metrics

# Configuration du logger
setup_logging(level="INFO", log_format="text")
logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """
    Définit le seed pour la reproductibilité.
    
    Args:
        seed: Valeur du seed
    """
    random.seed(seed)
    np.random.seed(seed)


def load_data(cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données train et validation.
    
    Args:
        cfg: Configuration Hydra
    
    Returns:
        Tuple (train_df, val_df)
    """
    logger.info("Chargement des données...")
    
    train_path = Path(cfg.paths.processed_dir) / "train.csv"
    val_path = Path(cfg.paths.processed_dir) / "val.csv"
    
    train_df = load_csv(train_path)
    val_df = load_csv(val_path)
    
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")
    
    return train_df, val_df


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, URLFeatureExtractor]:
    """
    Prépare les features pour l'entraînement.
    
    Args:
        train_df: DataFrame d'entraînement
        val_df: DataFrame de validation
        cfg: Configuration Hydra
    
    Returns:
        Tuple (X_train, X_val, y_train, y_val, feature_extractor)
    """
    logger.info("Extraction des features...")
    
    text_col = cfg.data.text_column
    target_col = cfg.data.target_column
    
    # Extraire X et y
    X_train_urls = train_df[text_col]
    y_train = train_df[target_col].values
    
    X_val_urls = val_df[text_col]
    y_val = val_df[target_col].values
    
    # Initialiser le feature extractor
    feature_extractor = URLFeatureExtractor(cfg.features)
    
    # Entraîner et transformer
    logger.info("Entraînement du feature extractor sur train set...")
    X_train = feature_extractor.fit_transform(X_train_urls)
    
    logger.info("Transformation du validation set...")
    X_val = feature_extractor.transform(X_val_urls)
    
    logger.info(f"Shape X_train: {X_train.shape}")
    logger.info(f"Shape X_val: {X_val.shape}")
    
    # Afficher la distribution des classes
    train_dist = pd.Series(y_train).value_counts()
    val_dist = pd.Series(y_val).value_counts()
    logger.info(f"Distribution train: {train_dist.to_dict()}")
    logger.info(f"Distribution val: {val_dist.to_dict()}")
    
    return X_train, X_val, y_train, y_val, feature_extractor


def create_optuna_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Any,
) -> callable:
    """
    Crée la fonction objectif pour Optuna.
    
    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        cfg: Configuration Hydra
    
    Returns:
        Fonction objectif Optuna
    """
    def objective(trial: optuna.Trial) -> float:
        """
        Fonction objectif Optuna: teste un ensemble d'hyperparamètres.
        
        Args:
            trial: Trial Optuna
        
        Returns:
            Score de validation (à maximiser)
        """
        # Suggérer les hyperparamètres depuis l'espace de recherche
        params = {}
        
        optuna_space = cfg.model.default.optuna_space
        
        for param_name, param_config in optuna_space.items():
            if param_config["type"] == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=True,
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"],
                )
        
        # Ajouter les paramètres fixes depuis la config
        fixed_params = {
            "random_state": cfg.train.default.seed,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        params.update(fixed_params)
        
        # Créer et évaluer le modèle avec cross-validation
        model = LogisticRegression(**params)
        
        # Cross-validation stratifiée
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cfg.train.default.cv.n_splits,
            scoring="f1",
            n_jobs=-1,
        )
        
        # Retourner le score moyen
        return cv_scores.mean()
    
    return objective


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Any,
) -> dict[str, Any]:
    """
    Optimise les hyperparamètres avec Optuna.
    
    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        cfg: Configuration Hydra
    
    Returns:
        Meilleurs hyperparamètres trouvés
    """
    logger.info("=" * 80)
    logger.info("OPTIMISATION DES HYPERPARAMÈTRES AVEC OPTUNA")
    logger.info("=" * 80)
    
    # Créer l'étude Optuna
    study = optuna.create_study(
        direction=cfg.train.default.optuna.direction,
        study_name="phishing_detection_optimization",
    )
    
    # Créer la fonction objectif
    objective = create_optuna_objective(X_train, y_train, cfg)
    
    # Callback MLflow pour logger les trials
    mlflc = MLflowCallback(
        tracking_uri=cfg.mlflow.tracking_uri,
        metric_name="cv_f1_score",
    )
    
    # Optimiser
    logger.info(f"Lancement de {cfg.train.default.optuna.n_trials} trials...")
    study.optimize(
        objective,
        n_trials=cfg.train.default.optuna.n_trials,
        n_jobs=cfg.train.default.optuna.n_jobs,
        show_progress_bar=cfg.train.default.optuna.show_progress_bar,
        callbacks=[mlflc],
    )
    
    # Afficher les résultats
    logger.info("=" * 80)
    logger.info("RÉSULTATS DE L'OPTIMISATION")
    logger.info("=" * 80)
    logger.info(f"Meilleur score: {study.best_value:.4f}")
    logger.info(f"Meilleurs paramètres: {study.best_params}")
    logger.info("=" * 80)
    
    return study.best_params


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict[str, Any],
    cfg: Any,
) -> LogisticRegression:
    """
    Entraîne le modèle final avec les meilleurs hyperparamètres.
    
    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        best_params: Meilleurs hyperparamètres d'Optuna
        cfg: Configuration Hydra
    
    Returns:
        Modèle entraîné
    """
    logger.info("Entraînement du modèle final...")
    
    # Préparer les paramètres finaux
    final_params = {
        **best_params,
        "random_state": cfg.train.default.seed,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    
    # Créer et entraîner le modèle
    model = LogisticRegression(**final_params)
    model.fit(X_train, y_train)
    
    logger.info("✓ Modèle entraîné")
    
    return model


def evaluate_model(
    model: LogisticRegression,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """
    Évalue le modèle sur le validation set.
    
    Args:
        model: Modèle entraîné
        X_val: Features de validation
        y_val: Labels de validation
    
    Returns:
        Dictionnaire de métriques
    """
    logger.info("Évaluation sur validation set...")
    
    # Prédictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    
    # Calculer les métriques
    metrics = calculate_metrics(y_val, y_pred, y_proba)
    
    # Afficher les métriques
    print_metrics(metrics, title="Métriques Validation Set")
    
    return metrics


def save_artifacts(
    model: LogisticRegression,
    feature_extractor: URLFeatureExtractor,
    cfg: Any,
) -> dict[str, Path]:
    """
    Sauvegarde le modèle et les artifacts.
    
    Args:
        model: Modèle entraîné
        feature_extractor: Feature extractor entraîné
        cfg: Configuration Hydra
    
    Returns:
        Dictionnaire des chemins sauvegardés
    """
    logger.info("Sauvegarde des artifacts...")
    
    models_dir = Path(cfg.paths.models_dir)
    ensure_dir(models_dir)
    
    artifacts = {}
    
    # Sauvegarder le modèle
    model_path = models_dir / cfg.model.default.save_name
    save_joblib(model, model_path)
    artifacts["model"] = model_path
    logger.info(f"✓ Modèle sauvegardé: {model_path}")
    
    # Sauvegarder les vectorizers
    if feature_extractor.tfidf_word is not None:
        tfidf_word_path = models_dir / cfg.model.default.vectorizers.tfidf_word
        save_joblib(feature_extractor.tfidf_word, tfidf_word_path)
        artifacts["tfidf_word"] = tfidf_word_path
        logger.info(f"✓ TF-IDF mots sauvegardé: {tfidf_word_path}")
    
    if feature_extractor.tfidf_char is not None:
        tfidf_char_path = models_dir / cfg.model.default.vectorizers.tfidf_char
        save_joblib(feature_extractor.tfidf_char, tfidf_char_path)
        artifacts["tfidf_char"] = tfidf_char_path
        logger.info(f"✓ TF-IDF char sauvegardé: {tfidf_char_path}")
    
    if feature_extractor.scaler is not None:
        scaler_path = models_dir / cfg.model.default.vectorizers.scaler
        save_joblib(feature_extractor.scaler, scaler_path)
        artifacts["scaler"] = scaler_path
        logger.info(f"✓ Scaler sauvegardé: {scaler_path}")
    
    # Sauvegarder la configuration
    config_path = models_dir / "config.yaml"
    save_config(cfg, config_path)
    artifacts["config"] = config_path
    logger.info(f"✓ Config sauvegardée: {config_path}")
    
    return artifacts


def log_to_mlflow(
    model: LogisticRegression,
    best_params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, Path],
    cfg: Any,
) -> None:
    """
    Log l'expérience dans MLflow.
    
    Args:
        model: Modèle entraîné
        best_params: Meilleurs hyperparamètres
        metrics: Métriques de validation
        artifacts: Chemins des artifacts sauvegardés
        cfg: Configuration Hydra
    """
    logger.info("Logging dans MLflow...")
    
    # Configurer MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        # Logger les hyperparamètres
        mlflow.log_params(best_params)
        
        # Logger les paramètres de config importants
        mlflow.log_param("dataset", cfg.data.kaggle_dataset)
        mlflow.log_param("text_column", cfg.data.text_column)
        mlflow.log_param("target_column", cfg.data.target_column)
        mlflow.log_param("seed", cfg.train.default.seed)
        mlflow.log_param("n_trials", cfg.train.default.optuna.n_trials)
        
        # Logger les métriques
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Logger le modèle
        if cfg.mlflow.log_models:
            mlflow.sklearn.log_model(model, "model")
        
        # Logger les artifacts
        if cfg.mlflow.log_artifacts:
            for artifact_name, artifact_path in artifacts.items():
                mlflow.log_artifact(str(artifact_path), artifact_name)
        
        logger.info("✓ Expérience loggée dans MLflow")


def main() -> None:
    """
    Fonction principale du pipeline d'entraînement.
    """
    logger.info("=" * 80)
    logger.info("PIPELINE D'ENTRAÎNEMENT - NLP PHISHING DETECTION")
    logger.info("=" * 80)
    
    # Charger la configuration
    cfg = load_config()
    validate_config(cfg)
    print_config(cfg)
    
    # Définir le seed
    set_seed(cfg.train.default.seed)
    logger.info(f"Seed: {cfg.train.default.seed}")
    
    # Charger les données
    train_df, val_df = load_data(cfg)
    
    # Préparer les features
    X_train, X_val, y_train, y_val, feature_extractor = prepare_features(
        train_df, val_df, cfg
    )
    
    # Optimiser les hyperparamètres
    best_params = optimize_hyperparameters(X_train, y_train, cfg)
    
    # Entraîner le modèle final
    model = train_final_model(X_train, y_train, best_params, cfg)
    
    # Évaluer le modèle
    metrics = evaluate_model(model, X_val, y_val)
    
    # Sauvegarder les artifacts
    artifacts = save_artifacts(model, feature_extractor, cfg)
    
    # Logger dans MLflow
    log_to_mlflow(model, best_params, metrics, artifacts, cfg)
    
    logger.info("=" * 80)
    logger.info("✓ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()