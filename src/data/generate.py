"""Génération d'un jeu de données synthétique de churn.

Choix pédagogique : des données synthétiques (make_classification) plutôt qu'un
dataset téléchargé. Avantages — aucun téléchargement, 100% reproductible, et
surtout on peut INJECTER un décalage de distribution (`drift_shift`) pour
démontrer concrètement le monitoring de drift et le ré-entraînement.

Point clé : la RÉFÉRENCE (données d'entraînement/baseline) et le COURANT
(données "de production") sont tirés du MÊME processus génératif (même
random_state, puis découpe). Ainsi, sans drift, un modèle entraîné sur la
référence généralise au courant ; avec `drift_shift` > 0, les performances
chutent ET la dérive est détectée — une histoire cohérente.

Produit :
  data/reference/reference.csv  -> baseline (sert à l'entraînement)
  data/raw/current.csv          -> "production" (éventuellement driftée)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.config import ROOT, load_params

FEATURE_PREFIX = "feature_"
TARGET = "churn"


def _feature_names(n: int) -> list[str]:
    return [f"{FEATURE_PREFIX}{i}" for i in range(n)]


def _apply_drift(df: pd.DataFrame, shift: float, seed: int) -> pd.DataFrame:
    """Décale la moyenne d'un tiers des features pour simuler une dérive."""
    if not shift:
        return df
    cols = [c for c in df.columns if c != TARGET]
    rng = np.random.default_rng(seed)
    drifted = rng.choice(cols, size=max(1, len(cols) // 3), replace=False)
    df = df.copy()
    df[drifted] = df[drifted] + shift
    return df


def make_dataset(
    params: dict, drift_shift: float = 0.0, n_samples: int | None = None
) -> pd.DataFrame:
    """Un seul jeu de données (utilisé en tests et pour la découpe de main())."""
    d = params["data"]
    X, y = make_classification(
        n_samples=n_samples or d["n_samples"],
        n_features=d["n_features"],
        n_informative=d["n_informative"],
        n_redundant=2,
        weights=[1 - d["churn_rate"], d["churn_rate"]],
        class_sep=d["class_sep"],
        flip_y=0.03,
        random_state=d["random_state"],
    )
    df = pd.DataFrame(X, columns=_feature_names(d["n_features"]))
    df[TARGET] = y
    df = _apply_drift(df, drift_shift, seed=d["random_state"])
    return df


def make_reference_and_current(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Référence et courant issus du MÊME processus (découpe d'un pool commun)."""
    d = params["data"]
    n = d["n_samples"]
    pool = make_dataset(params, drift_shift=0.0, n_samples=2 * n)
    pool = pool.sample(frac=1.0, random_state=d["random_state"]).reset_index(drop=True)

    reference = pool.iloc[:n].reset_index(drop=True)
    current = pool.iloc[n:].reset_index(drop=True)

    # La dérive ne s'applique qu'au courant (la référence reste la baseline).
    current = _apply_drift(current, float(d.get("drift_shift", 0.0)), seed=d["random_state"] + 1)
    return reference, current


def main() -> None:
    params = load_params()
    (ROOT / "data" / "reference").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)

    reference, current = make_reference_and_current(params)
    ref_path = ROOT / "data" / "reference" / "reference.csv"
    cur_path = ROOT / "data" / "raw" / "current.csv"
    reference.to_csv(ref_path, index=False)
    current.to_csv(cur_path, index=False)

    shift = float(params["data"].get("drift_shift", 0.0))
    print(f"[data] référence -> {Path(ref_path).relative_to(ROOT)}  ({len(reference)} lignes)")
    print(
        f"[data] courant   -> {Path(cur_path).relative_to(ROOT)}  "
        f"({len(current)} lignes, drift={shift})"
    )


if __name__ == "__main__":
    main()
