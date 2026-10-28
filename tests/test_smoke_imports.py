"""Smoke tests : les nouveaux modules s'importent (couvre promotion + orchestration)."""
from __future__ import annotations


def test_promote_imports():
    from src.pipelines import promote

    assert hasattr(promote, "main")


def test_retrain_flow_imports():
    from src.pipelines import retrain_flow

    # Le flux Prefect et ses tâches sont définis.
    assert hasattr(retrain_flow, "retrain_on_drift")
