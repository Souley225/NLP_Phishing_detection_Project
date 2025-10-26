"""
Téléchargement et préparation du dataset depuis Kaggle.

Ce script télécharge le dataset de phishing URLs depuis Kaggle,
effectue un nettoyage minimal et prépare les splits train/val/test.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import os
import sys
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import load_config, validate_config
from src.utils.io_utils import ensure_dir, save_csv
from src.utils.logging_utils import get_logger, setup_logging

# Configuration du logger
setup_logging(level="INFO", log_format="text")
logger = get_logger(__name__)


def download_from_kaggle(dataset_slug: str, download_path: Path) -> Path:
    """
    Télécharge un dataset depuis Kaggle.
    
    Args:
        dataset_slug: Identifiant Kaggle du dataset (ex: "username/dataset-name")
        download_path: Répertoire de destination
    
    Returns:
        Chemin du fichier téléchargé
    
    Raises:
        Exception: Si le téléchargement échoue
    """
    logger.info(f"Téléchargement du dataset Kaggle: {dataset_slug}")
    
    # Créer le répertoire de destination
    ensure_dir(download_path)
    
    # Initialiser l'API Kaggle
    api = KaggleApi()
    api.authenticate()
    
    try:
        # Télécharger le dataset
        api.dataset_download_files(
            dataset_slug,
            path=download_path,
            unzip=True,
        )
        
        logger.info(f"Dataset téléchargé dans: {download_path}")
        
        # Trouver le fichier CSV téléchargé
        csv_files = list(download_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {download_path}")
        
        # Retourner le premier fichier CSV trouvé
        return csv_files[0]
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {e}")
        raise


def clean_dataset(df: pd.DataFrame, text_col: str, target_col: str) -> pd.DataFrame:
    """
    Nettoie le dataset en supprimant les valeurs manquantes et doublons.
    
    Args:
        df: DataFrame à nettoyer
        text_col: Nom de la colonne contenant le texte (URL)
        target_col: Nom de la colonne cible
    
    Returns:
        DataFrame nettoyé
    """
    logger.info(f"Nettoyage du dataset - Taille initiale: {len(df)}")
    
    # Vérifier que les colonnes existent
    if text_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Colonnes manquantes. Attendues: {text_col}, {target_col}")
    
    # Supprimer les valeurs manquantes
    initial_size = len(df)
    df = df.dropna(subset=[text_col, target_col])
    logger.info(f"Lignes avec valeurs manquantes supprimées: {initial_size - len(df)}")
    
    # Supprimer les doublons
    initial_size = len(df)
    df = df.drop_duplicates(subset=[text_col])
    logger.info(f"Doublons supprimés: {initial_size - len(df)}")
    
    # Convertir les URLs en string et nettoyer les espaces
    df[text_col] = df[text_col].astype(str).str.strip()
    
    # Filtrer les URLs vides
    df = df[df[text_col].str.len() > 0]
    
    logger.info(f"Taille finale du dataset: {len(df)}")
    
    return df


def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divise le dataset en ensembles train/validation/test.
    
    Args:
        df: DataFrame à diviser
        target_col: Nom de la colonne cible
        test_size: Proportion du test set (0.2 = 20%)
        val_size: Proportion du validation set (0.1 = 10%)
        random_state: Seed pour reproductibilité
        stratify: Stratifier selon la cible pour préserver les proportions
    
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    logger.info("Division du dataset en train/val/test")
    
    # Stratification optionnelle
    stratify_col = df[target_col] if stratify else None
    
    # Split train/temp (temp = val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=stratify_col,
    )
    
    # Split temp en val/test
    val_ratio = val_size / (test_size + val_size)
    stratify_temp = temp_df[target_col] if stratify else None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=stratify_temp,
    )
    
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Afficher la distribution des classes
    for name, data in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        class_counts = data[target_col].value_counts()
        logger.info(f"{name} - Distribution: {class_counts.to_dict()}")
    
    return train_df, val_df, test_df


def main() -> None:
    """
    Fonction principale pour télécharger et préparer le dataset.
    """
    logger.info("=" * 80)
    logger.info("TÉLÉCHARGEMENT ET PRÉPARATION DU DATASET")
    logger.info("=" * 80)
    
    # Charger la configuration
    cfg = load_config()
    validate_config(cfg)
    
    # Paramètres depuis la config
    dataset_slug = cfg.data.kaggle_dataset
    text_col = cfg.data.text_column
    target_col = cfg.data.target_column
    test_size = cfg.data.test_size
    val_size = cfg.data.val_size
    stratify = cfg.data.stratify
    
    raw_dir = Path(cfg.paths.raw_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    
    # Créer les répertoires
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)
    
    # Télécharger le dataset
    try:
        csv_path = download_from_kaggle(dataset_slug, raw_dir)
    except Exception as e:
        logger.error(f"Échec du téléchargement: {e}")
        logger.info("Vérifiez vos credentials Kaggle (~/.kaggle/kaggle.json)")
        sys.exit(1)
    
    # Charger le dataset
    logger.info(f"Chargement du fichier: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset chargé - Shape: {df.shape}")
    logger.info(f"Colonnes: {list(df.columns)}")
    
    # Nettoyer le dataset
    df_clean = clean_dataset(df, text_col, target_col)
    
    # Diviser en train/val/test
    train_df, val_df, test_df = split_dataset(
        df_clean,
        target_col=target_col,
        test_size=test_size,
        val_size=val_size,
        random_state=42,
        stratify=stratify,
    )
    
    # Sauvegarder les splits
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"
    
    save_csv(train_df, train_path)
    save_csv(val_df, val_path)
    save_csv(test_df, test_path)
    
    logger.info(f"Train sauvegardé: {train_path}")
    logger.info(f"Val sauvegardé: {val_path}")
    logger.info(f"Test sauvegardé: {test_path}")
    
    logger.info("=" * 80)
    logger.info("✓ TÉLÉCHARGEMENT ET PRÉPARATION TERMINÉS")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()