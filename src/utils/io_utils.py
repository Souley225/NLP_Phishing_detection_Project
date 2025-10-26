"""
Utilitaires pour la lecture et l'écriture de fichiers.

Ce module fournit des fonctions pour gérer les opérations I/O
sur différents formats (pickle, joblib, JSON, CSV).

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_pickle(obj: Any, path: Path | str) -> None:
    """
    Sauvegarde un objet Python avec pickle.
    
    Args:
        obj: Objet à sauvegarder
        path: Chemin du fichier de sortie
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path | str) -> Any:
    """
    Charge un objet Python depuis un fichier pickle.
    
    Args:
        path: Chemin du fichier pickle
    
    Returns:
        Objet Python chargé
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {path}")
    
    with open(path, "rb") as f:
        return pickle.load(f)


def save_joblib(obj: Any, path: Path | str, compress: int = 3) -> None:
    """
    Sauvegarde un objet avec joblib (optimisé pour les modèles sklearn).
    
    Args:
        obj: Objet à sauvegarder (modèle, vectorizer, etc.)
        path: Chemin du fichier de sortie
        compress: Niveau de compression (0-9, 3 par défaut)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(obj, path, compress=compress)


def load_joblib(path: Path | str) -> Any:
    """
    Charge un objet depuis un fichier joblib.
    
    Args:
        path: Chemin du fichier joblib
    
    Returns:
        Objet chargé
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {path}")
    
    return joblib.load(path)


def save_json(data: dict | list, path: Path | str, indent: int = 2) -> None:
    """
    Sauvegarde des données en JSON.
    
    Args:
        data: Dictionnaire ou liste à sauvegarder
        path: Chemin du fichier de sortie
        indent: Niveau d'indentation pour la lisibilité
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Path | str) -> dict | list:
    """
    Charge des données depuis un fichier JSON.
    
    Args:
        path: Chemin du fichier JSON
    
    Returns:
        Données chargées (dict ou list)
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path | str, index: bool = False) -> None:
    """
    Sauvegarde un DataFrame pandas en CSV.
    
    Args:
        df: DataFrame à sauvegarder
        path: Chemin du fichier de sortie
        index: Sauvegarder l'index ou non
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=index, encoding="utf-8")


def load_csv(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """
    Charge un DataFrame pandas depuis un CSV.
    
    Args:
        path: Chemin du fichier CSV
        **kwargs: Arguments additionnels pour pd.read_csv
    
    Returns:
        DataFrame chargé
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {path}")
    
    return pd.read_csv(path, **kwargs)


def ensure_dir(path: Path | str) -> Path:
    """
    Crée un répertoire s'il n'existe pas.
    
    Args:
        path: Chemin du répertoire
    
    Returns:
        Chemin du répertoire (Path object)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: Path | str, pattern: str = "*") -> list[Path]:
    """
    Liste tous les fichiers d'un répertoire correspondant à un pattern.
    
    Args:
        directory: Chemin du répertoire
        pattern: Pattern de recherche (ex: "*.pkl", "*.csv")
    
    Returns:
        Liste des chemins de fichiers trouvés
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    return list(directory.glob(pattern))