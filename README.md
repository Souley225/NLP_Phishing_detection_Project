# PhishGuard : Détection de Phishing par NLP

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com/)
[![HF Space](https://img.shields.io/badge/Demo-HF%20Space-FFD21E?logo=huggingface&logoColor=black)](https://sallsou-nlp-phishing-detection.hf.space)
[![License](https://img.shields.io/badge/License-MIT-22c55e?logo=opensourceinitiative&logoColor=white)](LICENSE)

---

Chaque jour, des millions de personnes reçoivent des liens frauduleux par email ou SMS : des URLs conçues pour ressembler à de vrais sites (banques, réseaux sociaux, services en ligne) dans le but de voler des identifiants ou des données personnelles. C'est ce qu'on appelle le **phishing**.

**PhishGuard** est un outil qui analyse une URL en moins d'une seconde et répond à une question simple : ce lien est-il dangereux ?

Pas besoin de cliquer dessus. L'outil examine uniquement la structure du lien (sa longueur, ses caractères, son domaine, sa profondeur) et donne un verdict avec un niveau de confiance. Aucune connexion externe n'est effectuée.

> Essayez la démo : [sallsou-nlp-phishing-detection.hf.space](https://sallsou-nlp-phishing-detection.hf.space)

---

### Sous le capot

PhishGuard est un projet de Machine Learning entraîné sur **549 000 URLs réelles** (phishing et légitimes). Il atteint un **F1-Score de 95.5%**, ce qui signifie qu'il détecte correctement 19 liens sur 20, dans les deux sens : sans trop accuser des sites innocents, sans laisser passer des liens dangereux.

---

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Pipeline NLP](#pipeline-nlp)
- [Architecture système](#architecture-système)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API](#api)
- [Performances](#performances)
- [Déploiement](#déploiement)
- [Stack technique](#stack-technique)
- [Structure du projet](#structure-du-projet)
- [Dataset](#dataset)
- [Contact](#contact)

---

## Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| **Analyse autonome** | Basée uniquement sur la chaîne URL, sans appel DNS ni réseau |
| **Normalisation URL** | Canonicalisation automatique (scheme, trailing slash) pour résultats cohérents |
| **Seuil optimal** | Threshold calibré sur la validation (0.51) pour maximiser le F1 |
| **API REST** | Endpoint FastAPI pour intégration programmatique |
| **Interface mobile** | Dashboard Streamlit mobile-first pour tests manuels |
| **CI/CD** | Déploiement automatique via GitHub Actions vers HF Space |

---

## Pipeline NLP

Diagramme interactif : [Voir sur Excalidraw](https://excalidraw.com/#json=neIVqIaK1OPgjBa4hevYi,ZHXNqRdOpOxseFj2re--qw)

```
URL saisie
    │
    ▼
_normalize_url()          # Ajout https://, canonicalisation slash final
    │
    ├──────────────────────────────────────────┐────────────────────────┐
    ▼                                          ▼                        ▼
TF-IDF Mots                          TF-IDF Caractères         Features Lexicales
bigrammes [1,2]                      4-grammes [2,4]           16 features
20 000 features                      15 000 features           (longueur, entropie,
                                                                IP, TLD, sous-domaines…)
    │                                          │                        │
    └──────────────────────────────────────────┘────────────────────────┘
                                    │
                                    ▼
                        scipy.sparse.hstack
                        35 016 features (sparse CSR)
                                    │
                                    ▼
                    CalibratedClassifierCV
                    LinearSVC (C=0.5, max_iter=2000)
                    calibration isotonic sur validation
                                    │
                              proba_phishing
                                    │
                         seuil optimal : 0.51
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
                  PHISHING                    LEGITIME
```

**Features lexicales extraites :**

| Feature | Description |
|---------|-------------|
| `url_length` | Longueur totale de l'URL |
| `entropy` | Entropie de Shannon (URLs aléatoires ont une entropie élevée) |
| `has_ip` | Présence d'une adresse IP dans le domaine |
| `suspicious_tld` | TLD dans une liste noire (.tk, .ml, .ga, .cf…) |
| `subdomain_count` | Nombre de sous-domaines |
| `path_depth` | Profondeur du chemin (`/a/b/c` = 3) |
| `digit_ratio` | Proportion de chiffres dans l'URL |
| `num_at`, `num_question`, `num_ampersand`, `num_equal` | Caractères spéciaux suspects |

---

## Architecture système

Diagramme interactif : [Voir sur Excalidraw](https://excalidraw.com/#json=-lLaTNyKAGgYyJrXUNoeJ,02Jbhm79Op7Z5YtX2vnLBg)

```
                        ┌─────────────────────────────────┐
                        │         Utilisateur              │
                        └──────────────┬──────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                                     ▼
          Streamlit UI :8501                    FastAPI REST :8000
          (interface mobile)                   /predict  /health
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   ▼
                        ┌──────────────────┐
                        │  _normalize_url()│
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │ URLFeatureExtract│
                        │ TF-IDF + Lexical │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  LinearSVC +     │
                        │  Calibration     │
                        │  threshold=0.51  │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
                PHISHING                  LEGITIME

                        Artefacts partagés (models/)
                        ├── best_model.pkl      (789 KB)
                        ├── tfidf_word.pkl      (235 KB)
                        ├── tfidf_char.pkl      (164 KB)
                        ├── scaler.pkl
                        └── threshold.json

              GitHub master ──► GitHub Actions ──► HF Space (Docker)
```

---

## Installation

### Prérequis

- Python 3.11+
- Compte Kaggle (pour télécharger le dataset)
- Docker (optionnel, pour la prod)

### Étapes

```bash
# Cloner
git clone https://github.com/Souley225/NLP_Phishing_detection_Project.git
cd NLP_Phishing_detection_Project

# Environnement virtuel
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows

# Dépendances production
pip install -r requirements.txt

# Dépendances dev (tests, MLflow, Optuna…)
pip install -r requirements-dev.txt

# Configuration Kaggle
# Créer ~/.kaggle/kaggle.json avec {"username": "...", "key": "..."}
```

---

## Utilisation

### Téléchargement et préparation des données

```bash
python src/data/download_data.py     # Télécharge depuis Kaggle
python src/data/make_dataset.py      # Découpe train/val/test
```

### Réentraînement complet

```bash
python scripts/train_for_deploy.py
```

Ce script :
1. Charge `configs/config.yaml` et `configs/model/default.yaml` sans Hydra
2. Extrait les 35 016 features (TF-IDF word + char + lexicales)
3. Entraîne `LinearSVC` enveloppé dans `CalibratedClassifierCV`
4. Recherche le seuil optimal sur la validation
5. Sauvegarde tous les artefacts dans `models/`

### Lancer les services

```bash
# API FastAPI
uvicorn src.serving.api:app --port 8000

# Interface Streamlit
streamlit run src/ui/app.py --server.port 8501

# Docker (API + UI ensemble)
docker compose up --build
```

### Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

---

## API

Base URL : `http://localhost:8000`

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Vérification de l'état du service |
| POST | `/predict` | Prédiction sur une URL |
| GET | `/docs` | Documentation Swagger interactive |

### Exemple

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-verify.tk/account/confirm"}'
```

**Réponse :**
```json
{
  "url": "https://paypal-verify.tk/account/confirm",
  "prediction": 1,
  "label": "phishing",
  "proba_phishing": 0.97,
  "proba_legitimate": 0.03
}
```

---

## Performances

Évaluées sur le jeu de test (20% du dataset, 109 807 URLs), seuil calibré à **0.51** :

| Métrique | Score |
|----------|-------|
| **F1-Score** | **95.5%** |
| **Précision** (phishing) | **96.7%** |
| **Rappel** (phishing) | **94.3%** |
| Amélioration vs baseline | +2.85 pts F1 |

**Comparaison des approches testées :**

| Modèle | Features | F1 | Notes |
|--------|----------|----|-------|
| LogisticRegression | 8k | 92.6% | Baseline |
| RandomForest 300 arbres | 80k | N/A | OOM : matrice 355k x 80k trop dense |
| **LinearSVC + CalibratedCV** | **35k** | **95.5%** | Actuel, efficace sur sparse |

**Latence (CPU, sans GPU) :**
- 1 URL : < 5 ms
- 1 000 URLs en batch : < 500 ms

> Le modèle analyse uniquement la syntaxe de l'URL. Des patterns de contournement (URLs légitimes avec beaucoup de paramètres, nouveaux TLDs) restent des limites connues du jeu de données Kaggle utilisé.

---

## Déploiement

### Docker local

```bash
docker compose up --build
# API  → http://localhost:8000
# UI   → http://localhost:8501
```

### Hugging Face Space (production)

Le déploiement est entièrement automatisé via **GitHub Actions** :

```
git push origin master
    │
    ▼
.github/workflows/  ──►  huggingface_hub.upload_folder()
                              │
                              ▼
                    HF Space (Docker SDK)
                    https://sallsou-nlp-phishing-detection.hf.space
```

Le workflow push le code source ET les artefacts `models/*.pkl` + `models/threshold.json` vers le Space. Le Dockerfile reconstruit l'image à chaque push.

Pour déclencher manuellement un redéploiement des artefacts après réentraînement :

```bash
python scripts/upload_to_hf.py   # ou via GitHub Actions "Run workflow"
```

---

## Stack technique

| Catégorie | Technologies |
|-----------|--------------|
| **ML/NLP** | scikit-learn, LinearSVC, CalibratedClassifierCV, TF-IDF, scipy.sparse |
| **Feature engineering** | tldextract, numpy, pandas |
| **API** | FastAPI, Uvicorn, Pydantic |
| **UI** | Streamlit (mobile-first) |
| **DevOps** | Docker, Docker Compose, GitHub Actions, HF Space |
| **Config** | Hydra (expérimentation), YAML natif (déploiement) |
| **Qualité** | pytest, ruff |

---

## Structure du projet

```
.
├── configs/
│   ├── config.yaml            # Config principale (features, paths, data)
│   ├── model/default.yaml     # LinearSVC params
│   └── train/default.yaml     # Seed, epochs
├── models/                    # Artefacts entraînés
│   ├── best_model.pkl         # CalibratedClassifierCV(LinearSVC)
│   ├── tfidf_word.pkl         # Vectorizer mots (20k bigrammes)
│   ├── tfidf_char.pkl         # Vectorizer chars (15k 4-grammes)
│   ├── scaler.pkl             # StandardScaler (features lexicales)
│   └── threshold.json         # Seuil optimal { "threshold": 0.51, "f1": 0.955 }
├── scripts/
│   ├── train_for_deploy.py    # Réentraînement sans MLflow/Hydra
│   └── e2e_test_deployed.py   # Tests E2E sur le Space déployé
├── src/
│   ├── data/                  # Acquisition et split des données
│   ├── features/              # URLFeatureExtractor (TF-IDF + lexical)
│   ├── models/                # Entraînement, évaluation (avec MLflow/Optuna)
│   ├── serving/               # API FastAPI
│   ├── ui/                    # Interface Streamlit
│   └── utils/                 # io_utils, metrics, logging
├── tests/                     # Tests unitaires
├── Dockerfile
├── docker-compose.yml
├── requirements.txt           # Dépendances production
├── requirements-dev.txt       # Dépendances dev (MLflow, Optuna, pytest…)
└── Makefile
```

---

## Dataset

Source : [Kaggle — Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls) (taruntiwarihp)

| Statistique | Valeur |
|-------------|--------|
| Total URLs | 549 346 |
| Phishing | ~274 000 (50%) |
| Légitimes | ~275 000 (50%) |
| Train | 355 036 |
| Validation | 50 719 |
| Test | 109 807 (20%) |

---

## Contact

**Souleymane Sall**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-souleymanes--sall-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/souleymanes-sall/)
[![GitHub](https://img.shields.io/badge/GitHub-Souley225-181717?logo=github&logoColor=white)](https://github.com/Souley225)
[![Email](https://img.shields.io/badge/Email-sallsouleymane2207%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:sallsouleymane2207@gmail.com)

---

## Ressources

**Dataset**
- [Phishing Site URLs sur Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)

**Articles de référence**
- [BERT for Phishing Detection](https://www.sciencedirect.com/science/article/pii/S1877050921014368)
- [URL-based Features for Phishing Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC8935623/)
- [Hybrid NLP Approach](https://www.mdpi.com/2079-9292/11/22/3647)

**Documentation**
- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn — LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [scikit-learn — CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- [Streamlit](https://docs.streamlit.io/)

---

*Dernière mise à jour : avril 2026*
