# NLP_phishing_detection_Project - Dockerfile
# Auteur: Souleymane Sall
# Email: sallsouleymane2207@gmail.com

# Image de base Python 3.11 slim (légère)
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Souleymane Sall <sallsouleymane2207@gmail.com>"
LABEL description="NLP Phishing Detection API & UI"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copier le code source
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p data/raw data/processed data/predictions models logs

# Exposer les ports (FastAPI: 8000, Streamlit: 8501)
EXPOSE 8000 8501

# Healthcheck pour l'API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Commande par défaut (peut être overridée par docker-compose ou Render)
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]