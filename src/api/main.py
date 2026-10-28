"""API de serving FastAPI.

Expose :
  GET  /health   -> état + modèle chargé
  POST /predict  -> prédiction de churn (journalisée en base)
  GET  /metrics  -> métriques Prometheus (instrumentation automatique)

Le modèle est chargé une seule fois au démarrage depuis le Model Registry
MLflow (repli local), et conservé dans app.state.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.orm import Session

from src.api.model_loader import ModelBundle, load_model
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.config import load_params
from src.db.connexion import get_db
from src.db.crud import init_db, log_prediction

PARAMS = load_params()
FEATURE_NAMES = [f"feature_{i}" for i in range(PARAMS["data"]["n_features"])]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Démarrage ---
    try:
        init_db()
    except Exception as e:  # noqa: BLE001 — la base peut être absente en dev
        print(f"[api] init_db ignoré ({e.__class__.__name__}: {e})")
    try:
        app.state.bundle = load_model()
        print(f"[api] modèle chargé : source={app.state.bundle.source} "
              f"version={app.state.bundle.version}")
    except Exception as e:  # noqa: BLE001
        app.state.bundle = None
        print(f"[api] AUCUN modèle chargé ({e}). /predict renverra 503.")
    yield
    # --- Arrêt (rien à nettoyer) ---


app = FastAPI(title="Churn MLOps API", version="0.1.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def get_bundle() -> ModelBundle:
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return bundle


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = getattr(app.state, "bundle", None)
    return HealthResponse(
        status="ok",
        model_loaded=bundle is not None,
        model_source=bundle.source if bundle else None,
        model_version=bundle.version if bundle else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    db: Session = Depends(get_db),
    bundle: ModelBundle = Depends(get_bundle),
) -> PredictResponse:
    missing = [f for f in FEATURE_NAMES if f not in req.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Features manquantes : {missing[:5]}{'...' if len(missing) > 5 else ''}",
        )
    row = pd.DataFrame([[req.features[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

    proba = float(bundle.model.predict_proba(row)[0, 1])
    pred = int(proba >= 0.5)

    try:
        log_prediction(
            db,
            features=req.features,
            prediction=pred,
            probability=proba,
            model_source=bundle.source,
            model_version=bundle.version,
        )
    except Exception as e:  # noqa: BLE001 — la journalisation ne doit pas casser la prédiction
        db.rollback()
        print(f"[api] log_prediction échoué ({e.__class__.__name__})")

    return PredictResponse(
        churn=pred,
        probability=proba,
        model_source=bundle.source,
        model_version=bundle.version,
    )
