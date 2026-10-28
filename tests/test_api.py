"""Tests de l'API de serving (FastAPI TestClient)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import FEATURE_NAMES, app


def test_health_ok():
    with TestClient(app):
        client = TestClient(app)
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_response(local_model):
    payload = {"features": {f: 0.1 for f in FEATURE_NAMES}}
    with TestClient(app) as client:  # déclenche le lifespan -> chargement modèle
        r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["churn"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_source"] in ("mlflow", "local")


def test_predict_rejects_missing_features(local_model):
    payload = {"features": {"feature_0": 0.1}}  # incomplet
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_metrics_endpoint_exposed(local_model):
    with TestClient(app) as client:
        r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_request" in r.text
