"""Schémas Pydantic v2 d'entrée/sortie de l'API."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Vecteur de features d'un client. Doit contenir feature_0 .. feature_{n-1}."""

    features: dict[str, float] = Field(
        ...,
        examples=[{"feature_0": 0.1, "feature_1": -1.2, "feature_2": 0.5}],
        description="Dictionnaire {nom_de_feature: valeur}.",
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # autorise model_source/model_version

    churn: int = Field(..., description="1 = churn prédit, 0 = non-churn")
    probability: float = Field(..., ge=0.0, le=1.0)
    model_source: str
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_source: str | None = None
    model_version: str | None = None
