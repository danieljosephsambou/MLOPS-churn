"""Préparation des données : chargement et split train/test.

Minimal ici (les données synthétiques sont déjà propres), mais c'est l'endroit
où brancheraient, sur un vrai projet, l'imputation, l'encodage, le scaling,
le feature engineering — idéalement dans un Pipeline sklearn versionné.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ROOT
from src.data.generate import TARGET


def _load(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable — lance d'abord `make data`.")
    return pd.read_csv(path)


def load_reference() -> pd.DataFrame:
    """Données baseline servant à l'entraînement."""
    return _load(ROOT / "data" / "reference" / "reference.csv")


def load_current() -> pd.DataFrame:
    """Données "de production" servant à l'évaluation et au monitoring."""
    return _load(ROOT / "data" / "raw" / "current.csv")


def split(df: pd.DataFrame, test_size: float, random_state: int):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
