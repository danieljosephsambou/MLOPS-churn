"""Évaluation du modèle servi sur les données courantes.

Permet de vérifier qu'un modèle déjà entraîné tient toujours ses performances
sur de nouvelles données (base d'un contrôle de régression de modèle en CI/CD).
"""
from __future__ import annotations

from src.api.model_loader import load_model
from src.config import load_params
from src.data.preprocess import load_current, split
from src.pipelines.train import evaluate as _evaluate


def main() -> dict[str, float]:
    params = load_params()
    bundle = load_model()
    df = load_current()
    _, X_test, _, y_test = split(
        df, params["data"]["test_size"], params["data"]["random_state"]
    )
    metrics = _evaluate(bundle.model, X_test, y_test)
    print(f"[evaluate] source={bundle.source} version={bundle.version} -> {metrics}")
    return metrics


if __name__ == "__main__":
    main()
