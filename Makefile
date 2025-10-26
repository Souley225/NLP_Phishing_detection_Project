# NLP_phishing_detection_Project - Makefile
# Auteur: Souleymane Sall
# Email: sallsouleymane2207@gmail.com

.PHONY: help setup download train evaluate predict serve ui docker-build docker-up docker-down test lint format clean

# Variables
PYTHON := python
PIP := pip
VENV := .venv
STREAMLIT_PORT := 8501
API_PORT := 8000

# Couleurs pour l'affichage
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Afficher l'aide
	@echo "$(GREEN)NLP_phishing_detection_Project - Commandes disponibles$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Installer les dépendances
	@echo "$(GREEN)Installation des dépendances...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Installation terminée$(NC)"

download: ## Télécharger le dataset depuis Kaggle
	@echo "$(GREEN)Téléchargement du dataset...$(NC)"
	$(PYTHON) src/data/make_dataset.py
	@echo "$(GREEN)✓ Dataset téléchargé$(NC)"

train: ## Entraîner le modèle (Hydra + Optuna + MLflow)
	@echo "$(GREEN)Entraînement du modèle...$(NC)"
	$(PYTHON) src/models/train.py
	@echo "$(GREEN)✓ Entraînement terminé$(NC)"

evaluate: ## Évaluer le modèle entraîné
	@echo "$(GREEN)Évaluation du modèle...$(NC)"
	$(PYTHON) src/models/evaluate.py
	@echo "$(GREEN)✓ Évaluation terminée$(NC)"

predict: ## Faire des prédictions (batch ou unitaire)
	@echo "$(GREEN)Prédiction...$(NC)"
	@echo "$(YELLOW)Usage: make predict INPUT='http://example.com'$(NC)"
	$(PYTHON) src/models/predict.py --text "$(INPUT)"

serve: ## Lancer l'API FastAPI
	@echo "$(GREEN)Lancement de l'API FastAPI sur http://localhost:$(API_PORT)$(NC)"
	uvicorn src.serving.api:app --reload --host 0.0.0.0 --port $(API_PORT)

ui: ## Lancer l'interface Streamlit
	@echo "$(GREEN)Lancement de l'interface Streamlit sur http://localhost:$(STREAMLIT_PORT)$(NC)"
	streamlit run src/ui/app.py --server.port $(STREAMLIT_PORT) --server.address 0.0.0.0

mlflow: ## Lancer MLflow UI
	@echo "$(GREEN)Lancement de MLflow UI sur http://localhost:5000$(NC)"
	mlflow ui --host 0.0.0.0 --port 5000

docker-build: ## Builder l'image Docker
	@echo "$(GREEN)Build de l'image Docker...$(NC)"
	docker build -t phishing-detector:latest .
	@echo "$(GREEN)✓ Image buildée$(NC)"

docker-up: ## Lancer les services avec Docker Compose
	@echo "$(GREEN)Lancement des services Docker...$(NC)"
	docker compose up --build
	@echo "$(GREEN)✓ Services lancés$(NC)"
	@echo "$(YELLOW)API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)UI:  http://localhost:8501$(NC)"

docker-down: ## Arrêter les services Docker
	@echo "$(GREEN)Arrêt des services Docker...$(NC)"
	docker compose down
	@echo "$(GREEN)✓ Services arrêtés$(NC)"

test: ## Lancer les tests
	@echo "$(GREEN)Lancement des tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "$(GREEN)✓ Tests terminés$(NC)"

test-fast: ## Lancer les tests (sans coverage)
	@echo "$(GREEN)Tests rapides...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests terminés$(NC)"

lint: ## Vérifier la qualité du code
	@echo "$(GREEN)Vérification du code avec ruff...$(NC)"
	ruff check src/ tests/
	@echo "$(GREEN)Vérification des types avec mypy...$(NC)"
	mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✓ Vérification terminée$(NC)"

format: ## Formatter le code
	@echo "$(GREEN)Formatage du code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)✓ Formatage terminé$(NC)"

clean: ## Nettoyer les fichiers temporaires
	@echo "$(GREEN)Nettoyage...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ htmlcov/ .tox/ .nox/
	@echo "$(GREEN)✓ Nettoyage terminé$(NC)"

clean-data: ## Supprimer les données téléchargées
	@echo "$(RED)Suppression des données...$(NC)"
	rm -rf data/raw/* data/processed/*
	@echo "$(GREEN)✓ Données supprimées$(NC)"

clean-models: ## Supprimer les modèles entraînés
	@echo "$(RED)Suppression des modèles...$(NC)"
	rm -rf models/*.pkl models/*.joblib
	@echo "$(GREEN)✓ Modèles supprimés$(NC)"

clean-mlflow: ## Supprimer les expériences MLflow
	@echo "$(RED)Suppression des expériences MLflow...$(NC)"
	rm -rf mlruns/ mlartifacts/
	@echo "$(GREEN)✓ MLflow nettoyé$(NC)"

clean-all: clean clean-data clean-models clean-mlflow ## Nettoyage complet
	@echo "$(GREEN)✓ Nettoyage complet terminé$(NC)"

install: setup download ## Installation complète (deps + data)
	@echo "$(GREEN)✓ Installation complète terminée$(NC)"

pipeline: download train evaluate ## Pipeline complet (download → train → evaluate)
	@echo "$(GREEN)✓ Pipeline complet terminé$(NC)"

# Workflow local complet
local: pipeline serve ## Workflow local: pipeline + serve API

# Workflow production
prod: docker-build docker-up ## Build et lance en production locale (Docker)