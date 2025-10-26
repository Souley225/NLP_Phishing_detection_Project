"""
Calcul des métriques de performance pour la classification.

Ce module fournit des fonctions pour calculer et afficher
les métriques d'évaluation des modèles de classification.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calcule toutes les métriques de classification.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions du modèle
        y_proba: Probabilités prédites (optionnel, pour ROC-AUC)
    
    Returns:
        Dictionnaire contenant toutes les métriques
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }
    
    # Ajouter ROC-AUC si les probabilités sont disponibles
    if y_proba is not None:
        try:
            # Pour la classification binaire, utiliser les probas de la classe positive
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
            
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba_pos))
        except ValueError:
            # Si ROC-AUC ne peut pas être calculé (ex: une seule classe)
            metrics["roc_auc"] = 0.0
    
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> str:
    """
    Génère un rapport de classification détaillé.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions du modèle
        target_names: Noms des classes (optionnel)
    
    Returns:
        Rapport de classification formaté
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de confusion.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions du modèle
    
    Returns:
        Matrice de confusion (array 2D)
    """
    return confusion_matrix(y_true, y_pred)


def calculate_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Calcule les métriques par classe.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions du modèle
    
    Returns:
        Dictionnaire avec les métriques pour chaque classe
    """
    # Calculer precision, recall, f1 pour chaque classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    
    # Organiser les résultats par classe
    classes = np.unique(y_true)
    metrics_by_class = {}
    
    for i, class_label in enumerate(classes):
        metrics_by_class[str(class_label)] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }
    
    return metrics_by_class


def print_metrics(metrics: dict[str, float], title: str = "Métriques") -> None:
    """
    Affiche les métriques de manière formatée.
    
    Args:
        metrics: Dictionnaire de métriques
        title: Titre à afficher
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")
    
    for metric_name, value in metrics.items():
        metric_label = metric_name.replace("_", " ").title()
        print(f"{metric_label:<30} : {value:>8.4f}")
    
    print(f"{'=' * 60}\n")


def format_metrics_for_mlflow(metrics: dict[str, float]) -> dict[str, float]:
    """
    Formate les métriques pour le logging MLflow.
    
    Ajoute un préfixe pour organiser les métriques dans l'UI MLflow.
    
    Args:
        metrics: Dictionnaire de métriques
    
    Returns:
        Dictionnaire formaté pour MLflow
    """
    return {f"metrics/{key}": value for key, value in metrics.items()}