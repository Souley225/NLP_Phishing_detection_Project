# NLP_phishing_detection_Project - Dockerfile (HF Spaces)
# Auteur: Souleymane Sall
# Email: sallsouleymane2207@gmail.com
#
# Single process: Streamlit boots FastAPI in a background thread.
# HF Spaces port: 7860

FROM python:3.11-slim

LABEL maintainer="Souleymane Sall <sallsouleymane2207@gmail.com>"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    API_URL=http://127.0.0.1:8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed data/predictions logs

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "src/ui/app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false"]
