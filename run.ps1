# run.ps1 — équivalent Windows du Makefile (les commandes passent par docker compose).
# Usage :  .\run.ps1 <cible>
# Exemple : .\run.ps1 up   puis   .\run.ps1 pipeline

param([Parameter(Position = 0)][string]$Target = "help")

$ErrorActionPreference = "Stop"

function Run($cmd) {
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
}

$RUN = "docker compose run --rm mlflow"

switch ($Target) {
    "up"       { Run "docker compose up -d --build" }
    "down"     { Run "docker compose down" }
    "ps"       { Run "docker compose ps" }
    "data"     { Run "$RUN python -m src.data.generate" }
    "train"    { Run "$RUN python -m src.pipelines.train" }
    "evaluate" { Run "$RUN python -m src.pipelines.evaluate" }
    "drift"    { Run "$RUN python -m src.monitoring.drift" }
    "promote"  { Run "$RUN python -m src.pipelines.promote" }
    "retrain"  { Run "$RUN python -m src.pipelines.retrain_flow" }
    "reload"   { Run "docker compose restart api" }
    "test"     { Run "$RUN pytest" }
    "logs"     { Run "docker compose logs -f api" }
    "pipeline" {
        # Pipeline complet : données -> entraînement -> promotion -> rechargement API
        Run "$RUN python -m src.data.generate"
        Run "$RUN python -m src.pipelines.train"
        Run "$RUN python -m src.pipelines.promote"
        Run "docker compose restart api"
    }
    "demo-drift" {
        # Active la dérive, régénère, puis lance le flux de ré-entraînement automatique
        (Get-Content params.yaml) -replace 'drift_shift: 0.0', 'drift_shift: 2.0' | Set-Content params.yaml
        Run "$RUN python -m src.pipelines.retrain_flow"
        Run "docker compose restart api"
        Write-Host "Pense a remettre drift_shift a 0.0 dans params.yaml." -ForegroundColor Yellow
    }
    default {
        Write-Host "Cibles disponibles :" -ForegroundColor Green
        Write-Host "  up         Construire et demarrer toute la stack"
        Write-Host "  down       Arreter la stack"
        Write-Host "  ps         Etat des services"
        Write-Host "  data       Generer les jeux de donnees"
        Write-Host "  train      Entrainer + enregistrer dans MLflow"
        Write-Host "  evaluate   Evaluer le modele servi sur les donnees courantes"
        Write-Host "  drift      Generer le rapport de derive (Evidently)"
        Write-Host "  promote    Promouvoir le meilleur modele en production"
        Write-Host "  retrain    Flux Prefect : detecter la derive et re-entrainer si besoin"
        Write-Host "  reload     Recharger le modele dans l'API"
        Write-Host "  test       Lancer les tests"
        Write-Host "  logs       Suivre les logs de l'API"
        Write-Host "  pipeline   data -> train -> promote -> reload"
        Write-Host "  demo-drift Activer la derive et declencher le re-entrainement auto"
    }
}
