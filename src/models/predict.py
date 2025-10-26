"""
Inférence batch et unitaire pour la détection de phishing.

Ce script permet de faire des prédictions sur de nouvelles URLs,
soit en mode batch (fichier CSV) soit en mode unitaire (URL unique).

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import load_config, validate_config
from src.features.build_features import URLFeatureExtractor
from src.utils.io_utils import load_csv, load_joblib, save_csv
from src.utils.logging_utils import get_logger, setup_logging

# Configuration du logger
setup_logging(level="INFO", log_format="text")
logger = get_logger(__name__)


def load_model_artifacts(cfg: any) -> tuple:
    """
    Charge le modèle et les feature extractors.
    
    Args:
        cfg: Configuration Hydra
    
    Returns:
        Tuple (model, feature_extractor)
    """
    models_dir = Path(cfg.paths.models_dir)
    
    # Charger le modèle
    model = load_joblib(models_dir / cfg.model.default.save_name)
    
    # Reconstruire le feature extractor
    feature_extractor = URLFeatureExtractor(cfg.features)
    
    if cfg.features.tfidf_word.use:
        feature_extractor.tfidf_word = load_joblib(
            models_dir / cfg.model.default.vectorizers.tfidf_word
        )
    
    if cfg.features.tfidf_char.use:
        feature_extractor.tfidf_char = load_joblib(
            models_dir / cfg.model.default.vectorizers.tfidf_char
        )
    
    if cfg.features.lexical.use:
        feature_extractor.scaler = load_joblib(
            models_dir / cfg.model.default.vectorizers.scaler
        )
    
    feature_extractor.is_fitted = True
    
    return model, feature_extractor


def predict_single(url: str, model: any, feature_extractor: URLFeatureExtractor) -> dict:
    """
    Prédit si une URL est du phishing ou légitime.
    
    Args:
        url: URL à analyser
        model: Modèle entraîné
        feature_extractor: Feature extractor
    
    Returns:
        Dictionnaire avec la prédiction et la confiance
    """
    # Transformer l'URL en features
    X = feature_extractor.transform(pd.Series([url]))
    
    # Prédiction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Préparer le résultat
    result = {
        "url": url,
        "prediction": int(prediction),
        "label": "phishing" if prediction == 1 else "legitimate",
        "confidence": float(probabilities[prediction]),
        "proba_legitimate": float(probabilities[0]),
        "proba_phishing": float(probabilities[1]),
    }
    
    return result


def predict_batch(
    input_path: Path,
    output_path: Path,
    model: any,
    feature_extractor: URLFeatureExtractor,
    url_column: str = "URL",
) -> None:
    """
    Prédit sur un fichier CSV d'URLs.
    
    Args:
        input_path: Chemin du fichier CSV d'entrée
        output_path: Chemin du fichier CSV de sortie
        model: Modèle entraîné
        feature_extractor: Feature extractor
        url_column: Nom de la colonne contenant les URLs
    """
    logger.info(f"Chargement du fichier: {input_path}")
    df = load_csv(input_path)
    
    if url_column not in df.columns:
        raise ValueError(f"Colonne '{url_column}' introuvable. Colonnes: {list(df.columns)}")
    
    logger.info(f"Prédiction sur {len(df)} URLs...")
    
    # Transformer les features
    X = feature_extractor.transform(df[url_column])
    
    # Prédictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Ajouter les résultats au DataFrame
    df["prediction"] = predictions
    df["label"] = ["phishing" if p == 1 else "legitimate" for p in predictions]
    df["confidence"] = [probabilities[i][p] for i, p in enumerate(predictions)]
    df["proba_legitimate"] = probabilities[:, 0]
    df["proba_phishing"] = probabilities[:, 1]
    
    # Sauvegarder
    save_csv(df, output_path)
    logger.info(f"✓ Prédictions sauvegardées: {output_path}")
    
    # Afficher un résumé
    summary = df["label"].value_counts()
    logger.info(f"\nRésumé des prédictions:\n{summary}")


def main() -> None:
    """
    Fonction principale pour l'inférence.
    """
    parser = argparse.ArgumentParser(description="Prédiction de phishing sur URLs")
    parser.add_argument(
        "--text",
        type=str,
        help="URL unique à analyser",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Fichier CSV d'URLs à analyser",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Fichier CSV de sortie (pour mode batch)",
    )
    parser.add_argument(
        "--url-column",
        type=str,
        default="URL",
        help="Nom de la colonne contenant les URLs (défaut: URL)",
    )
    
    args = parser.parse_args()
    
    # Vérifier qu'au moins un mode est spécifié
    if not args.text and not args.input:
        parser.error("Spécifiez soit --text pour une URL unique, soit --input pour un fichier CSV")
    
    # Charger la configuration
    logger.info("Chargement de la configuration...")
    cfg = load_config()
    validate_config(cfg)
    
    # Charger le modèle et les extractors
    logger.info("Chargement du modèle...")
    model, feature_extractor = load_model_artifacts(cfg)
    logger.info("✓ Modèle chargé")
    
    # Mode unitaire
    if args.text:
        logger.info(f"\nAnalyse de l'URL: {args.text}")
        result = predict_single(args.text, model, feature_extractor)
        
        print("\n" + "=" * 80)
        print("RÉSULTAT DE LA PRÉDICTION")
        print("=" * 80)
        print(f"URL         : {result['url']}")
        print(f"Prédiction  : {result['label'].upper()}")
        print(f"Confiance   : {result['confidence']:.2%}")
        print(f"Prob Légitime : {result['proba_legitimate']:.2%}")
        print(f"Prob Phishing : {result['proba_phishing']:.2%}")
        print("=" * 80)
    
    # Mode batch
    if args.input:
        input_path = Path(args.input)
        
        if not input_path.exists():
            logger.error(f"Fichier introuvable: {input_path}")
            sys.exit(1)
        
        # Déterminer le chemin de sortie
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(cfg.paths.predictions_dir) / "predictions.csv"
        
        predict_batch(
            input_path,
            output_path,
            model,
            feature_extractor,
            url_column=args.url_column,
        )


if __name__ == "__main__":
    main()