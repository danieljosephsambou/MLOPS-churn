"""Pipeline d'entraînement.

Étapes : charge les données -> split -> entraîne -> évalue -> QUALITY GATE
-> log dans MLflow (params, métriques, modèle) -> enregistre dans le Model
Registry. Un repli local (joblib) est toujours écrit pour que l'API puisse
servir même sans serveur MLflow (utile en dev / CI).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import ROOT, get_settings, load_params
from src.data.preprocess import load_current, split


def build_model(cfg: dict):
    if cfg["type"] == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=cfg["random_state"])
    return RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        random_state=cfg["random_state"],
        n_jobs=-1,
    )


def evaluate(model, X_test, y_test) -> dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def _save_local(model, metrics: dict) -> Path:
    settings = get_settings()
    out = ROOT / settings.local_model_path
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return out


def _log_to_mlflow(model, params: dict, metrics: dict, X_test) -> None:
    """Best-effort : si le serveur MLflow est injoignable, on n'échoue pas."""
    settings = get_settings()
    # Doit être posé avant l'import de mlflow/GitPython pour couper le warning Git.
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.models.signature import infer_signature

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run():
            mlflow.log_params(params["model"])
            mlflow.log_params({f"data_{k}": v for k, v in params["data"].items()})
            mlflow.log_metrics(metrics)
            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                registered_model_name=settings.model_name,
            )
        print(f"[mlflow] run loggé + modèle '{settings.model_name}' enregistré "
              f"({settings.mlflow_tracking_uri})")
    except Exception as e:  # noqa: BLE001 — on veut un repli robuste en dev/CI
        print(f"[mlflow] indisponible ({e.__class__.__name__}: {e}). "
              f"Repli : modèle sauvegardé localement uniquement.")


def main() -> dict[str, float]:
    params = load_params()
    # On entraîne sur les données "courantes" (le dernier lot de production
    # étiqueté). Ainsi, quand la distribution dérive, un ré-entraînement adapte
    # réellement le modèle à la nouvelle réalité — c'est la boucle MLOps.
    df = load_current()
    X_train, X_test, y_train, y_test = split(
        df, params["data"]["test_size"], params["data"]["random_state"]
    )

    model = build_model(params["model"])
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    print(f"[train] métriques : {metrics}")

    # --- Quality gate : on refuse de promouvoir un modèle trop faible ---
    threshold = params["training"]["min_roc_auc"]
    if metrics["roc_auc"] < threshold:
        raise SystemExit(
            f"[train] ÉCHEC quality gate : roc_auc={metrics['roc_auc']:.3f} "
            f"< seuil {threshold}. Modèle NON promu."
        )

    local_path = _save_local(model, metrics)
    print(f"[train] modèle local -> {local_path.relative_to(ROOT)}")
    _log_to_mlflow(model, params, metrics, X_test)
    return metrics


if __name__ == "__main__":
    main()
