.PHONY: help install data train evaluate drift promote retrain serve test lint up down logs clean

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Installe les dépendances Python
	pip install -r requirements.txt

data:  ## Génère les jeux de données (référence + courant)
	python -m src.data.generate

train:  ## Entraîne le modèle et l'enregistre dans MLflow
	python -m src.pipelines.train

evaluate:  ## Évalue le dernier modèle enregistré
	python -m src.pipelines.evaluate

drift:  ## Génère le rapport de drift (Evidently) dans reports/
	python -m src.monitoring.drift

promote:  ## Promeut le meilleur modèle en production (champion/challenger)
	python -m src.pipelines.promote

retrain:  ## Flux Prefect : générer -> détecter dérive -> ré-entraîner si besoin
	python -m src.pipelines.retrain_flow

serve:  ## Lance l'API en local (hors docker)
	uvicorn src.api.main:app --reload --port 8000

test:  ## Lance les tests
	pytest

lint:  ## Lint du code
	ruff check .

up:  ## Démarre toute la stack (docker-compose)
	docker compose up -d --build

down:  ## Arrête la stack
	docker compose down

logs:  ## Suit les logs de l'API
	docker compose logs -f api

clean:  ## Nettoie artefacts locaux
	rm -rf data/raw/*.csv data/processed/*.csv data/reference/*.csv models/*.joblib reports/
