"""Configuration centralisée : variables d'environnement (.env) + params.yaml.

Un seul endroit pour la config = reproductibilité et pas de valeurs en dur
dispersées dans le code (un des principes de base du MLOps).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Paramètres d'infrastructure (lus depuis l'environnement / .env)."""

    # protected_namespaces=() : on autorise les champs `model_*` (model_name, model_stage)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # --- Base applicative (logs de prédictions) ---
    postgres_user: str = "mlops"
    postgres_password: str = "mlops"
    postgres_db: str = "mlops"
    postgres_host: str | None = None  # None => repli SQLite local (dev hors docker)
    postgres_port: int = 5432

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "churn"
    model_name: str = "churn-classifier"
    # Alias du registry à servir (ex. "production"). Si l'alias n'existe pas
    # encore, l'API sert automatiquement la dernière version enregistrée.
    model_alias: str = "production"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Repli local si le registry MLflow est indisponible ---
    local_model_path: str = "models/model.joblib"

    @property
    def database_url(self) -> str:
        """Postgres si POSTGRES_HOST est défini (docker), sinon SQLite local."""
        if self.postgres_host:
            return (
                f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return f"sqlite:///{ROOT / 'predictions.db'}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_params(path: str = "params.yaml") -> dict:
    """Charge les hyperparamètres du pipeline (versionnés avec le code)."""
    with open(ROOT / path) as f:
        return yaml.safe_load(f)
