"""Tests du modèle et du quality gate."""
from __future__ import annotations

import pytest

from src.config import load_params
from src.data.generate import make_dataset
from src.data.preprocess import split
from src.pipelines.train import build_model, evaluate


def _trained():
    params = load_params()
    df = make_dataset(params, drift_shift=0.0)
    X_train, X_test, y_train, y_test = split(df, 0.2, 42)
    model = build_model(params["model"]).fit(X_train, y_train)
    return model, X_test, y_test


def test_metrics_are_reported():
    model, X_test, y_test = _trained()
    metrics = evaluate(model, X_test, y_test)
    assert {"accuracy", "f1", "roc_auc"} <= metrics.keys()
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_model_beats_random():
    model, X_test, y_test = _trained()
    metrics = evaluate(model, X_test, y_test)
    # Un modèle entraîné doit battre nettement le hasard (AUC 0.5).
    assert metrics["roc_auc"] > 0.7


def test_quality_gate_blocks_weak_model(monkeypatch, tmp_path):
    """Si le seuil est inatteignable, le pipeline doit refuser de promouvoir."""
    import src.pipelines.train as train_mod

    base = load_params()
    params = {**base, "training": {"min_roc_auc": 0.999}}
    df = make_dataset(base, drift_shift=0.0)
    monkeypatch.setattr(train_mod, "load_params", lambda *a, **k: params)
    monkeypatch.setattr(train_mod, "load_current", lambda *a, **k: df)
    monkeypatch.setattr(train_mod, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        train_mod.main()
