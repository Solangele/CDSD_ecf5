# --- STAGE 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Empêche Python d'écrire des fichiers .pyc et d'utiliser un tampon pour les logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système nécessaires pour compiler certains packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Création du venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- STAGE 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Installation de curl pour le Healthcheck (souvent absent des versions slim)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CRUCIAL : Copier le package churnguard ET l'api
COPY . /app
COPY api/ ./api/

# On ajoute le répertoire courant au PYTHONPATH pour que "import churnguard" fonctionne
ENV PYTHONPATH="/app"

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]