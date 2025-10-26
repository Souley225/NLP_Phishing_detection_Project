"""
Configuration du système de logging structuré.

Ce module configure le logging en format JSON pour faciliter
l'analyse et le monitoring en production.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Formatter personnalisé pour générer des logs en format JSON.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formate un enregistrement de log en JSON.
        
        Args:
            record: Enregistrement de log à formater
        
        Returns:
            Chaîne JSON formatée
        """
        # Construction du dictionnaire de log
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Ajouter les informations d'exception si présentes
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Ajouter les attributs supplémentaires
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """
    Formatter pour logs en format texte lisible.
    """
    
    def __init__(self) -> None:
        """Initialise le formatter avec un format standard."""
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Path | str | None = None,
) -> logging.Logger:
    """
    Configure le système de logging pour l'application.
    
    Args:
        level: Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format des logs ("json" ou "text")
        log_file: Chemin optionnel du fichier de log
    
    Returns:
        Logger configuré
    """
    # Récupérer le logger root
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Supprimer les handlers existants
    logger.handlers.clear()
    
    # Choisir le formatter selon le format demandé
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    # Handler console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier (si spécifié)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Fichier = mode verbose
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Récupère un logger avec un nom spécifique.
    
    Args:
        name: Nom du logger (généralement __name__)
    
    Returns:
        Logger configuré
    """
    return logging.getLogger(name)


def log_dict(logger: logging.Logger, level: str, message: str, data: dict[str, Any]) -> None:
    """
    Log un message avec des données structurées additionnelles.
    
    Args:
        logger: Logger à utiliser
        level: Niveau de log (info, debug, warning, error)
        message: Message principal
        data: Données structurées à logger
    """
    # Créer un LogRecord avec des données extra
    log_method = getattr(logger, level.lower())
    log_method(message, extra={"extra_data": data})


def log_exception(logger: logging.Logger, message: str) -> None:
    """
    Log une exception avec sa stack trace complète.
    
    Args:
        logger: Logger à utiliser
        message: Message descriptif de l'erreur
    """
    logger.exception(message)


# Logger par défaut pour l'application
def get_default_logger() -> logging.Logger:
    """
    Récupère le logger par défaut de l'application.
    
    Returns:
        Logger configuré avec les paramètres par défaut
    """
    return setup_logging(
        level="INFO",
        log_format="json",
        log_file="logs/app.log",
    )