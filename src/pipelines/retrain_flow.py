"""Orchestration du ré-entraînement conditionnel à la dérive (Prefect).

Enchaîne automatiquement : générer les données courantes -> détecter la dérive
-> SI dérive significative : ré-entraîner le modèle puis le promouvoir
(champion/challenger). Sinon, ne rien faire.

Lancement local (sans serveur Prefect) :
    python -m src.pipelines.retrain_flow

En production, ce flux se planifierait (deployment Prefect) ou se déclencherait
sur alerte de monitoring.
"""
from __future__ import annotations

from prefect import flow, task
from prefect.logging import get_run_logger

from src.data import generate as data_gen
from src.monitoring import drift as drift_mod
from src.pipelines import promote as promote_mod
from src.pipelines import train as train_mod


@task(name="générer-données")
def generate_data() -> None:
    data_gen.main()


@task(name="détecter-dérive")
def detect_drift() -> dict:
    return drift_mod.main()


@task(name="ré-entraîner")
def retrain() -> dict:
    return train_mod.main()


@task(name="promouvoir")
def promote() -> dict:
    return promote_mod.main()


@flow(name="retrain-on-drift")
def retrain_on_drift(regenerate: bool = True) -> dict:
    logger = get_run_logger()

    if regenerate:
        generate_data()

    drift = detect_drift()
    if not drift["drift_detected"]:
        logger.info("Pas de dérive significative — aucun ré-entraînement nécessaire.")
        return {"retrained": False, "drift": drift}

    logger.info("Dérive détectée — ré-entraînement puis promotion du modèle.")
    metrics = retrain()
    decision = promote()
    return {"retrained": True, "drift": drift, "metrics": metrics, "promotion": decision}


if __name__ == "__main__":
    result = retrain_on_drift()
    print(f"[flow] résultat : retrained={result['retrained']}")
