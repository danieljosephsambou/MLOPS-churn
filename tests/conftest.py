"""Fixtures partagées : un modèle local est garanti pour les tests d'API."""
from __future__ import annotations

import os

import joblib
import pytest

# Forcer le mode dev local (SQLite + repli modèle local) avant tout import de config.
os.environ.pop("POSTGRES_HOST", None)
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5999"  # injoignable -> repli local
# MLflow injoignable : on coupe les retries/backoff pour que les tests ne pendent pas.
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"


@pytest.fixture(scope="session")
def local_model(tmp_path_factory):
    """Entraîne un petit modèle et l'écrit à l'emplacement du repli local."""
    from src.config import ROOT, get_settings, load_params
    from src.data.generate import make_dataset
    from src.data.preprocess import split
    from src.pipelines.train import build_model

    params = load_params()
    df = make_dataset(params, drift_shift=0.0)
    X_train, _, y_train, _ = split(df, 0.2, 42)
    model = build_model(params["model"])
    model.fit(X_train, y_train)

    path = ROOT / get_settings().local_model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
