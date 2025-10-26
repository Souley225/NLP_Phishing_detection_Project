# Détection de Phishing 

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-orange.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Un projet de détection automatique de sites de phishing en analysant simplement leurs URLs. Pas besoin de visiter le site ou d'utiliser des services externes.

**Par** : Souleymane Sall  
**Contact** : sallsouleymane2207@gmail.com  
**Date** : Octobre 2025

---

## C'est quoi ce projet ?

J'ai créé un système qui peut détecter si une URL est un site de phishing ou non, juste en regardant l'URL elle-même. Pas besoin de visiter le site, pas besoin de vérifier des listes noires, pas besoin de services externes.

Le modèle analyse l'URL comme du texte et repère les patterns suspects (domaines bizarres, caractères louches, structures anormales...).

**Résultats** : Plus de 97% de précision sur 549 000 URLs testées.

---

## Pourquoi ce projet ?

Les attaques de phishing sont partout. Les méthodes classiques pour les détecter ont des problèmes :
- Elles nécessitent des services externes (WHOIS, DNS...)
- Elles doivent charger les pages web (lent et risqué)
- Elles utilisent des listes noires (toujours en retard)

**Ma solution** : Analyser juste la chaîne de caractères de l'URL avec du NLP et du machine learning.

### Dataset utilisé

- **Source** : [Kaggle - Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
- **Taille** : ~549 000 URLs
- **Équilibré** : 50% phishing, 50% légitimes

---

## Comment ça marche ?

### Vue d'ensemble

```
URL brute → Extraction de caractéristiques → Modèle ML → Prédiction
```

J'utilise trois types de caractéristiques :

1. **Statistiques de l'URL** (8 features)
   - Longueur totale
   - Nombre de sous-domaines
   - Présence d'une adresse IP
   - Entropie (mesure du "désordre")
   - Proportion de chiffres
   - Nombre de caractères spéciaux (@, ?, &...)
   - TLD suspect (.tk, .ml...)
   - Profondeur des chemins

2. **Analyse des mots** (5000 features TF-IDF)
   - Je découpe l'URL en "mots" (paypal, login, secure...)
   - Le modèle apprend quels mots apparaissent dans le phishing

3. **Analyse des caractères** (3000 features TF-IDF)
   - Analyse des séquences de 2-5 caractères
   - Détecte les tentatives d'imitation (paypa1 vs paypal)

**Total** : Plus de 8000 caractéristiques analysées par URL.

### Le modèle

J'utilise une **régression logistique** optimisée avec :
- **Optuna** pour trouver les meilleurs hyperparamètres automatiquement
- **MLflow** pour suivre toutes les expériences et comparer les résultats
- **Validation croisée** pour éviter le surapprentissage

---

## Installation

### Ce dont vous avez besoin

- Python 3.11 ou plus récent
- Un compte Kaggle (pour télécharger le dataset)
- Docker (optionnel, pour le déploiement)

### Étapes d'installation

**1. Cloner le projet**
```bash
git clone https://github.com/Souley225/NLP_phishing_detection_Project.git
cd NLP_phishing_detection_Project
```

**2. Créer un environnement virtuel**
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Mac/Linux
# ou
.venv\Scripts\activate  # Sur Windows
```

**3. Installer les dépendances**
```bash
pip install -r requirements.txt
```

**4. Configurer Kaggle**

Créez le fichier `~/.kaggle/kaggle.json` avec vos identifiants :
```json
{
  "username": "votre_username",
  "key": "votre_cle_api"
}
```

Sur Mac/Linux, changez les permissions :
```bash
chmod 600 ~/.kaggle/kaggle.json
```

**5. Variables d'environnement**
```bash
cp .env.example .env
# Éditez .env avec vos infos
```

---

## Utilisation

### Le workflow complet en 5 commandes

```bash
# 1. Télécharger les données
make download

# 2. Entraîner le modèle
make train

# 3. Évaluer les performances
make evaluate

# 4. Lancer l'API
make serve

# 5. Lancer l'interface web (dans un autre terminal)
make ui
```

### Tester rapidement

```bash
# Prédire une seule URL
python src/models/predict.py --text "http://paypal-secure.tk/login.php"

# Prédire plusieurs URLs depuis un fichier CSV
python src/models/predict.py --input urls.csv --output predictions.csv
```

### Personnaliser l'entraînement

```bash
# Plus d'essais pour l'optimisation (meilleure performance)
python src/models/train.py train.n_trials=100

# Changer les paramètres du modèle
python src/models/train.py model.max_iter=300

# Utiliser un seed différent
python src/models/train.py train.seed=42
```

---

## L'API REST

Une fois lancée avec `make serve`, l'API est disponible sur `http://localhost:8000`.

### Vérifier que l'API fonctionne

```bash
curl http://localhost:8000/health
```

Réponse :
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-26T12:34:56"
}
```

### Analyser une URL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-verify.tk/account"}'
```

Réponse :
```json
{
  "url": "http://paypal-verify.tk/account",
  "prediction": 1,
  "label": "phishing",
  "confidence": 0.94,
  "timestamp": "2025-10-26T12:35:10"
}
```

La documentation interactive est disponible sur `http://localhost:8000/docs`.

---

## L'interface web

Lancez l'interface avec `make ui` et allez sur `http://localhost:8501`.

Vous pourrez :
- Entrer n'importe quelle URL pour l'analyser
- Voir immédiatement si c'est du phishing ou non
- Consulter le score de confiance
- Visualiser les mots/caractères qui ont influencé la décision
- Garder un historique de vos analyses

---

## Déploiement

### Avec Docker (local)

```bash
# Tout lancer en une commande
docker compose up --build

# L'API sera sur http://localhost:8000
# L'interface sur http://localhost:8501
```

### Sur le cloud (Render)

Le projet est configuré pour être déployé facilement sur Render :

1. Poussez votre code sur GitHub
2. Connectez votre repo sur [render.com](https://render.com)
3. Render détecte automatiquement le fichier `render.yaml`
4. Ajoutez vos variables d'environnement (KAGGLE_USERNAME, KAGGLE_KEY)
5. Cliquez sur "Deploy"

Le fichier `render.yaml` configure deux services :
- L'API FastAPI
- L'interface Streamlit

---

## Structure du projet

```
nlp-phishing-detector/
│
├── src/                       # Code source
│   ├── data/                  # Téléchargement et préparation des données
│   ├── features/              # Extraction des caractéristiques
│   ├── models/                # Entraînement et prédictions
│   ├── serving/               # API FastAPI
│   ├── ui/                    # Interface Streamlit
│   └── utils/                 # Fonctions utilitaires
│
├── configs/                   # Fichiers de configuration (Hydra)
├── tests/                     # Tests unitaires
├── data/                      # Données (pas sur Git)
├── models/                    # Modèles entraînés (pas sur Git)
├── mlruns/                    # Expériences MLflow (pas sur Git)
│
├── Dockerfile                 # Pour créer l'image Docker
├── docker-compose.yml         # Pour lancer les services localement
├── render.yaml                # Configuration du déploiement cloud
├── Makefile                   # Commandes rapides
└── requirements.txt           # Dépendances Python
```

---

## Suivi des expériences avec MLflow

MLflow garde une trace de tous mes essais d'entraînement.

### Lancer l'interface MLflow

```bash
mlflow ui
# Puis allez sur http://localhost:5000
```

Vous y verrez :
- Tous les paramètres testés
- Les performances de chaque essai
- Les graphiques (courbe ROC, matrice de confusion)
- Les modèles sauvegardés

C'est super pratique pour comparer différentes versions et choisir la meilleure.

---

## Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Tests d'un module spécifique
pytest tests/test_features.py -v

# Avec rapport de couverture
pytest tests/ --cov=src --cov-report=html
```

---

## Performances

Sur le jeu de test (données jamais vues pendant l'entraînement) :

| Métrique | Score |
|----------|-------|
| Précision globale | 97.2% |
| Précision (classe phishing) | 96.8% |
| Rappel (détection des phishing) | 97.6% |
| Score F1 | 97.2% |
| ROC-AUC | 99.1% |

**Vitesse** :
- Une URL : moins de 5ms
- 1000 URLs : moins de 500ms

---

## Stack technique

**Machine Learning & Data**
- scikit-learn (modèle et features)
- Optuna (optimisation automatique)
- pandas & numpy (manipulation des données)

**MLOps**
- MLflow (suivi des expériences)
- Hydra (configuration flexible)

**Web**
- FastAPI (API REST)
- Streamlit (interface utilisateur)
- Uvicorn (serveur)

**DevOps**
- Docker & Docker Compose
- Render (déploiement cloud)

**Qualité**
- pytest (tests)
- ruff, black, isort (formatage du code)

---

## Ce que j'ai appris

Ce projet m'a permis de mettre en pratique :

- Le **feature engineering** : transformer du texte brut en caractéristiques exploitables
- L'**optimisation d'hyperparamètres** : laisser l'algorithme trouver les meilleurs réglages
- Le **MLOps** : suivre mes expériences, versionner mes modèles
- Le **déploiement** : passer d'un notebook à une API production-ready
- Les **bonnes pratiques** : tests, documentation, configuration propre

---

## Améliorations possibles

Si je devais aller plus loin :

- Tester des modèles plus complexes (BERT, transformers)
- Ajouter des features temporelles (âge du domaine, historique)
- Créer un système d'apprentissage continu
- Ajouter une détection d'anomalies pour les nouveaux patterns
- Améliorer l'interface avec plus de visualisations
- Faire un monitoring en production (drift, performances)

---

## Contribution

Le projet est open source. Si vous voulez contribuer :

1. Fork le projet
2. Créez une branche pour votre feature
3. Faites vos modifications
4. Envoyez une pull request

Toute contribution est la bienvenue, que ce soit du code, de la doc, ou des idées !

---

## Licence

MIT License - Vous pouvez utiliser ce code librement.

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
