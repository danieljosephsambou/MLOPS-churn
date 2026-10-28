"""Chargement du modèle à servir.

Priorité au Model Registry MLflow (source de vérité en production) ; repli sur
le fichier joblib local pour que l'API démarre toujours en dev / CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib

from src.config import ROOT, get_settings


@dataclass
class ModelBundle:
    model: Any
    source: str   # "mlflow" | "local"
    version: str


def _load_from_mlflow() -> ModelBundle | None:
    settings = get_settings()
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()
        name = settings.model_name

        # 1) Priorité à l'alias (ex. "production"), s'il existe.
        alias = settings.model_alias
        if alias:
            try:
                mv = client.get_model_version_by_alias(name, alias)
                model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
                return ModelBundle(model=model, source="mlflow", version=f"{mv.version} (@{alias})")
            except Exception:  # noqa: BLE001 — alias absent : on sert la dernière version
                pass

        # 2) Sinon, la dernière version enregistrée.
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            return None
        latest = max(versions, key=lambda v: int(v.version))
        # flavor sklearn -> on récupère l'estimateur réel (predict + predict_proba)
        model = mlflow.sklearn.load_model(f"models:/{name}/{latest.version}")
        return ModelBundle(model=model, source="mlflow", version=str(latest.version))
    except Exception as e:  # noqa: BLE001
        print(f"[model_loader] MLflow indisponible ({e.__class__.__name__}). Repli local.")
        return None


def _load_local() -> ModelBundle:
    settings = get_settings()
    path = ROOT / settings.local_model_path
    if not path.exists():
        raise FileNotFoundError(
            f"Aucun modèle MLflow et pas de modèle local ({path}). "
            f"Lance `make data && make train`."
        )
    return ModelBundle(model=joblib.load(path), source="local", version="local")


def load_model() -> ModelBundle:
    return _load_from_mlflow() or _load_local()
