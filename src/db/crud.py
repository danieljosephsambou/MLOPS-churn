"""Modèle et accès aux logs de prédictions.

Chaque appel à /predict est journalisé : c'est la matière première du
monitoring (volume, dérive des entrées, distribution des sorties) et la base
d'un futur jeu de ré-entraînement.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, Session, mapped_column

from src.db.connexion import Base, engine


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.UTC)
    )
    features: Mapped[dict] = mapped_column(JSON)
    prediction: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    model_source: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def log_prediction(
    db: Session,
    *,
    features: dict,
    prediction: int,
    probability: float,
    model_source: str,
    model_version: str,
) -> PredictionLog:
    row = PredictionLog(
        features=features,
        prediction=prediction,
        probability=probability,
        model_source=model_source,
        model_version=model_version,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
