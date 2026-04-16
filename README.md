---
title: NLP Phishing Detection
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Detect phishing URLs with NLP (TF-IDF + Logistic Regression)
---

# Détection de Phishing 

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative&logoColor=white)](LICENSE)

Systeme de detection de phishing par analyse textuelle d'URLs utilisant le NLP et le Machine Learning. Precision de **97%+** sur 549 000 URLs.

---

## Table des matieres

- [Fonctionnalites](#fonctionnalites)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API](#api)
- [Deploiement](#deploiement)
- [Performances](#performances)
- [Stack technique](#stack-technique)
- [Contact](#contact)

---

## Fonctionnalites

| Fonctionnalite | Description |
|----------------|-------------|
| **Detection autonome** | Analyse basee uniquement sur la chaine URL, sans appel externe |
| **API REST** | Endpoint FastAPI pour integration |
| **Interface web** | Dashboard Streamlit pour tests manuels |
| **Tracking ML** | Suivi des experiences avec MLflow |
| **Optimisation auto** | Recherche d'hyperparametres via Optuna |
| **Deploiement cloud** | Configuration Render prete |

---

## Architecture

```
URL → Extraction Features → Modele ML → Prediction
         │
         ├── TF-IDF mots (5000 features)
         ├── TF-IDF caracteres (3000 features)
         └── Features lexicales (16 features)
```

**Features extraites :**
- Longueur, entropie, ratio chiffres
- Nombre de sous-domaines, profondeur du chemin
- Presence d'IP, TLD suspect
- Caracteres speciaux (@, ?, &, =)

---

## Installation

### Prerequis

- Python 3.11+
- Compte Kaggle (pour le dataset)
- Docker (optionnel)

### Etapes

```bash
# Cloner
git clone https://github.com/Souley225/NLP_phishing_detection_Project.git
cd NLP_phishing_detection_Project

# Environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Dependances
pip install -r requirements.txt

# Configuration Kaggle
# Creer ~/.kaggle/kaggle.json avec vos identifiants

# Variables d'environnement
cp .env.example .env
```

---

## Utilisation

### Commandes principales

```bash
make download    # Telecharger le dataset
make train       # Entrainer le modele
make evaluate    # Evaluer les performances
make serve       # Lancer l'API (port 8000)
make ui          # Lancer l'interface (port 8501)
make mlflow      # Lancer MLflow UI (port 5000)
```

### Prediction directe

```bash
# URL unique
python src/models/predict.py --text "http://paypal-secure.tk/login"

# Batch depuis fichier
python src/models/predict.py --input urls.csv --output predictions.csv
```

### Personnalisation

```bash
python src/models/train.py train.n_trials=100    # Plus d'iterations Optuna
python src/models/train.py model.max_iter=300    # Iterations modele
python src/models/train.py train.seed=42         # Seed custom
```

---

## API

Base URL : `http://localhost:8000`

### Endpoints

| Methode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Verification de sante |
| POST | `/predict` | Prediction sur URL |
| GET | `/docs` | Documentation Swagger |

### Exemple

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-verify.tk/account"}'
```

**Reponse :**
```json
{
  "url": "http://paypal-verify.tk/account",
  "prediction": 1,
  "label": "phishing",
  "proba_phishing": 0.94,
  "proba_legitimate": 0.06
}
```

---

## Deploiement

### Docker

```bash
# Local
docker compose up --build

# Services disponibles :
# API      → http://localhost:8000
# UI       → http://localhost:8501
# MLflow   → http://localhost:5000
```

### Render

1. Pousser le code sur GitHub
2. Connecter le repo sur [render.com](https://render.com)
3. Le fichier `render.yaml` configure automatiquement :
   - Service API (`phishing-api`)
   - Service UI (`phishing-ui`)
4. Ajouter les variables `KAGGLE_USERNAME` et `KAGGLE_KEY`

---

## Performances

| Metrique | Score |
|----------|-------|
| Precision globale | 97.2% |
| Precision (phishing) | 96.8% |
| Rappel (phishing) | 97.6% |
| F1-Score | 97.2% |
| ROC-AUC | 99.1% |

**Latence :**
- 1 URL : < 5ms
- 1000 URLs : < 500ms

---

## Stack technique

| Categorie | Technologies |
|-----------|--------------|
| **ML/NLP** | scikit-learn, Optuna, pandas, numpy |
| **MLOps** | MLflow, Hydra |
| **API** | FastAPI, Uvicorn, Pydantic |
| **UI** | Streamlit |
| **DevOps** | Docker, Docker Compose, Render |
| **Qualite** | pytest, ruff, black, mypy |

---

## Structure du projet

```
.
├── src/
│   ├── data/          # Acquisition des donnees
│   ├── features/      # Feature engineering
│   ├── models/        # Entrainement, evaluation, prediction
│   ├── serving/       # API FastAPI
│   ├── ui/            # Interface Streamlit
│   └── utils/         # Utilitaires
├── configs/           # Configuration Hydra
├── tests/             # Tests unitaires
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── Makefile
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v                              # Tests complets
pytest tests/ --cov=src --cov-report=html     # Avec couverture
```

---

## Dataset

Source : [Kaggle - Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
- 549 000 URLs
- 50% phishing, 50% legitimes

---

## Contact

**Souleymane Sall**  
📧 sallsouleymane2207@gmail.com  
💼 [LinkedIn](www.linkedin.com/in/souleymanes-sall)  
🐙 [Meduim](medium.com/@sallsouleymane66)

N'hésitez pas à me contacter si vous avez des questions ou des suggestions !

---

## Ressources

**Dataset**
- [Phishing Site URLs sur Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)

**Articles de recherche qui m'ont inspiré**
- [BERT for Phishing Detection](https://www.sciencedirect.com/science/article/pii/S1877050921014368)
- [URL-based Features](https://pmc.ncbi.nlm.nih.gov/articles/PMC8935623/)
- [Hybrid NLP Approach](https://www.mdpi.com/2079-9292/11/22/3647)

**Documentation**
- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)

---

*Ce README est régulièrement mis à jour. Dernière modification : Octobre 2025*
