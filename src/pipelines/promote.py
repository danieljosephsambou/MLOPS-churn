"""Promotion de modèle (champion / challenger) via les alias du Model Registry.

Compare la DERNIÈRE version enregistrée (challenger) au modèle actuellement en
production (alias "production", le champion), sur les données courantes. Si le
challenger est meilleur — ou s'il n'y a pas encore de champion — il est promu
en posant l'alias "production" sur sa version.

L'API sert l'alias "production" (voir model_loader) : la promotion prend donc
effet au prochain (re)chargement du modèle.
"""
from __future__ import annotations

import os

from src.config import get_settings, load_params
from src.data.preprocess import load_current, split
from src.pipelines.train import evaluate

# Métrique de décision pour départager champion et challenger.
DECISION_METRIC = "roc_auc"


def _score(client, model_uri: str) -> float:
    import mlflow.sklearn

    params = load_params()
    model = mlflow.sklearn.load_model(model_uri)
    df = load_current()
    _, X_test, _, y_test = split(df, params["data"]["test_size"], params["data"]["random_state"])
    return evaluate(model, X_test, y_test)[DECISION_METRIC]


def main() -> dict:
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    import mlflow
    from mlflow.tracking import MlflowClient

    settings = get_settings()
    alias = settings.model_alias or "production"
    name = settings.model_name

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise SystemExit(f"[promote] aucun modèle '{name}' enregistré — lance `train` d'abord.")
    challenger = max(versions, key=lambda v: int(v.version))

    # Champion actuel (s'il existe).
    try:
        champion = client.get_model_version_by_alias(name, alias)
    except Exception:  # noqa: BLE001
        champion = None

    challenger_score = _score(client, f"models:/{name}/{challenger.version}")

    if champion is None:
        client.set_registered_model_alias(name, alias, challenger.version)
        decision = {
            "promoted": True,
            "reason": "aucun champion existant",
            "new_production_version": challenger.version,
            "challenger_score": round(challenger_score, 4),
        }
    elif champion.version == challenger.version:
        decision = {
            "promoted": False,
            "reason": "le challenger est déjà en production",
            "production_version": champion.version,
        }
    else:
        champion_score = _score(client, f"models:/{name}@{alias}")
        if challenger_score >= champion_score:
            client.set_registered_model_alias(name, alias, challenger.version)
            decision = {
                "promoted": True,
                "reason": f"challenger {DECISION_METRIC}={challenger_score:.4f} "
                          f">= champion {champion_score:.4f}",
                "old_production_version": champion.version,
                "new_production_version": challenger.version,
            }
        else:
            decision = {
                "promoted": False,
                "reason": f"challenger {DECISION_METRIC}={challenger_score:.4f} "
                          f"< champion {champion_score:.4f}",
                "production_version": champion.version,
            }

    tag = "✅ PROMU" if decision["promoted"] else "⏸ conservé"
    print(f"[promote] {tag} — {decision['reason']}")
    print(f"[promote] détail : {decision}")
    return decision


if __name__ == "__main__":
    main()
