"""
Module de gestion de la configuration via Hydra.

Ce module charge et valide la configuration du projet en utilisant Hydra.
Toutes les configurations sont centralisées dans le dossier configs/.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_config(config_name: str = "config", overrides: list[str] | None = None) -> DictConfig:
    """
    Charge la configuration Hydra depuis le dossier configs/.
    
    Args:
        config_name: Nom du fichier de configuration principal (sans .yaml)
        overrides: Liste d'overrides Hydra optionnels
    
    Returns:
        Configuration Hydra chargée
    
    Exemples:
        >>> cfg = load_config()
        >>> cfg = load_config(overrides=["train.seed=999", "model.params.C=10.0"])
    """
    # Déterminer le chemin absolu du dossier configs
    config_dir = Path(__file__).parent.parent / "configs"
    config_dir = config_dir.resolve()
    
    # Initialiser Hydra avec le répertoire de configuration
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        # Composer la configuration avec les overrides
        cfg = compose(config_name=config_name, overrides=overrides or [])
    
    return cfg


def validate_config(cfg: DictConfig) -> None:
    """
    Valide la configuration chargée.
    
    Vérifie que tous les champs obligatoires sont présents et valides.
    
    Args:
        cfg: Configuration Hydra à valider
    
    Raises:
        ValueError: Si la configuration est invalide
    """
    # Vérifier les sections obligatoires
    required_sections = ["data", "paths", "features", "mlflow", "model", "train"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Section obligatoire manquante dans la config: {section}")
    
    # Vérifier les colonnes de données
    if not cfg.data.text_column or not cfg.data.target_column:
        raise ValueError("Les colonnes text_column et target_column doivent être définies")
    
    # Vérifier les chemins
    if not cfg.paths.data_dir:
        raise ValueError("Le chemin data_dir doit être défini")
    
    # Vérifier les paramètres de feature engineering
    if not any([cfg.features.tfidf_word.use, cfg.features.tfidf_char.use, cfg.features.lexical.use]):
        raise ValueError("Au moins une méthode de feature engineering doit être activée")


def get_config_value(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """
    Récupère une valeur de configuration avec gestion des clés imbriquées.
    
    Args:
        cfg: Configuration Hydra
        key: Clé au format "section.subsection.key"
        default: Valeur par défaut si la clé n'existe pas
    
    Returns:
        Valeur de configuration ou valeur par défaut
    
    Exemples:
        >>> value = get_config_value(cfg, "model.params.C", default=1.0)
    """
    try:
        keys = key.split(".")
        value = cfg
        for k in keys:
            value = value[k]
        return value
    except (KeyError, AttributeError):
        return default


def print_config(cfg: DictConfig) -> None:
    """
    Affiche la configuration de manière lisible.
    
    Args:
        cfg: Configuration Hydra à afficher
    """
    print("=" * 80)
    print("CONFIGURATION CHARGÉE")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)


def save_config(cfg: DictConfig, output_path: Path) -> None:
    """
    Sauvegarde la configuration dans un fichier YAML.
    
    Args:
        cfg: Configuration Hydra à sauvegarder
        output_path: Chemin du fichier de sortie
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        OmegaConf.save(cfg, f)