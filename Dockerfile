# NLP_phishing_detection_Project - Dockerfile (HF Spaces compatible)
# Auteur: Souleymane Sall
# Email: sallsouleymane2207@gmail.com
# Build: includes trained model artifacts (best_model.pkl, tfidf_word/char.pkl, scaler.pkl)
#
# Runs two services via startup.sh:
#   - FastAPI on port 8000 (internal)
#   - Streamlit on port 7860 (public, HF Spaces default)

FROM python:3.11-slim

LABEL maintainer="Souleymane Sall <sallsouleymane2207@gmail.com>"
LABEL description="NLP Phishing Detection API & UI"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    API_URL=http://localhost:8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed data/predictions logs && \
    chmod +x startup.sh

# HF Spaces expects port 7860
EXPOSE 7860 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["bash", "startup.sh"]
