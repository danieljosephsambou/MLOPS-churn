"""Rapport de dérive de données (data drift) avec Evidently.

Compare la distribution de RÉFÉRENCE (baseline d'entraînement) à la
distribution COURANTE (données de production simulées). Produit un rapport HTML
exploitable + un résumé JSON qui pourrait déclencher un ré-entraînement.

Démo : mettre `drift_shift` > 0 dans params.yaml, relancer `make data`, puis
`make drift` -> le rapport signale la dérive.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ROOT
from src.data.generate import TARGET

# Part de features driftées à partir de laquelle on déclenche une alerte /
# un ré-entraînement (plus sensible que le drapeau natif d'Evidently à 50%).
DRIFT_SHARE_THRESHOLD = 0.25


def _load(split: str) -> pd.DataFrame:
    mapping = {
        "reference": ROOT / "data" / "reference" / "reference.csv",
        "current": ROOT / "data" / "raw" / "current.csv",
    }
    path = mapping[split]
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable — lance d'abord `make data`.")
    return pd.read_csv(path).drop(columns=[TARGET], errors="ignore")


def main() -> dict:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    reference = _load("reference")
    current = _load("current")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    html_path = out_dir / "drift_report.html"
    report.save_html(str(html_path))

    summary = report.as_dict()["metrics"][0]["result"]
    n_total = summary.get("number_of_columns") or 0
    n_drifted = summary.get("number_of_drifted_columns") or 0
    share = summary.get("share_of_drifted_columns") or 0.0

    # Décision explicite et configurable : on ne se contente pas du drapeau
    # `dataset_drift` d'Evidently (vrai seulement si >50% des colonnes dérivent).
    # On déclenche dès qu'une part significative des features a dérivé.
    drift_detected = share >= DRIFT_SHARE_THRESHOLD

    result = {
        "evidently_dataset_drift": summary.get("dataset_drift"),  # drapeau natif (seuil 50%)
        "n_drifted_features": n_drifted,
        "n_total_features": n_total,
        "share_drifted": round(share, 3),
        "threshold": DRIFT_SHARE_THRESHOLD,
        "drift_detected": drift_detected,
    }
    (out_dir / "drift_summary.json").write_text(json.dumps(result, indent=2))

    print(f"[drift] rapport -> {Path(html_path).relative_to(ROOT)}")
    print(f"[drift] features driftées : {n_drifted}/{n_total} ({share:.0%})")
    if drift_detected:
        print(
            f"[drift] ⚠ DÉRIVE DÉTECTÉE (part {share:.0%} ≥ seuil {DRIFT_SHARE_THRESHOLD:.0%}) "
            f"— un ré-entraînement est recommandé."
        )
    else:
        print(f"[drift] ✓ pas de dérive significative (seuil {DRIFT_SHARE_THRESHOLD:.0%}).")
    return result


if __name__ == "__main__":
    main()
