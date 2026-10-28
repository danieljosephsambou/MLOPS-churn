"""Tests de la génération de données."""
from __future__ import annotations

from src.config import load_params
from src.data.generate import TARGET, make_dataset


def test_dataset_shape_and_columns():
    params = load_params()
    df = make_dataset(params, drift_shift=0.0)
    assert len(df) == params["data"]["n_samples"]
    assert TARGET in df.columns
    assert df.drop(columns=[TARGET]).shape[1] == params["data"]["n_features"]


def test_target_is_binary():
    params = load_params()
    df = make_dataset(params, drift_shift=0.0)
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_drift_shifts_distribution():
    params = load_params()
    base = make_dataset(params, drift_shift=0.0)
    drifted = make_dataset(params, drift_shift=3.0)
    # Au moins une feature doit voir sa moyenne nettement bouger.
    feats = [c for c in base.columns if c != TARGET]
    max_delta = max(abs(drifted[f].mean() - base[f].mean()) for f in feats)
    assert max_delta > 1.0
