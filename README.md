# MLOps churn — squelette pédagogique de bout en bout

Un squelette **MLOps complet et lançable**, 100 % open-source, tournant en local
via Docker Compose. Cas d'usage : prédiction de churn (classification binaire)
sur données tabulaires synthétiques.

> Ce projet montre *tous* les maillons d'une chaîne MLOps — suivi d'expériences,
> registry de modèles, serving, monitoring service, détection de drift, tests et
> CI — sur un modèle volontairement simple, pour se concentrer sur la plomberie.

L'architecture détaillée (composants, choix d'outils, diagramme) est dans
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Stack

scikit-learn · MLflow · FastAPI · PostgreSQL · Prometheus · Grafana · Evidently ·
Prefect · Docker Compose · GitHub Actions · pytest · ruff

## Windows

Windows n'a pas `make`. Un script **`run.ps1`** reproduit les cibles via Docker :

```powershell
.\run.ps1 up         # construire + démarrer toute la stack
.\run.ps1 pipeline   # data -> train -> promote -> recharger l'API
.\run.ps1 demo-drift # activer la dérive et déclencher le ré-entraînement auto
.\run.ps1            # liste toutes les cibles
```

Prérequis : WSL2 + Docker Desktop (moteur « Engine running »). Si `run.ps1` est
bloqué par la politique d'exécution : `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
Table de correspondance `make` → Windows en bas de ce fichier.

## Prérequis

- Docker + Docker Compose (pour la stack complète)
- Python 3.11+ (pour lancer le pipeline en local sans Docker)

## Démarrage rapide — tout en Docker

```bash
cp .env.example .env
make up            # démarre Postgres, MLflow, API, Prometheus, Grafana
```

Services exposés :

| Service | URL |
|---|---|
| API (OpenAPI) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

Entraîner un premier modèle (dans le conteneur d'entraînement) :

```bash
docker compose run --rm mlflow python -m src.data.generate
docker compose run --rm mlflow python -m src.pipelines.train
```

Le modèle est enregistré dans le Model Registry MLflow ; **redémarrez l'API**
(`docker compose restart api`) pour qu'elle le charge, puis testez :

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

## Démarrage rapide — en local, sans Docker

```bash
pip install -r requirements.txt
make data      # génère référence + courant
make train     # entraîne, quality gate, sauvegarde (repli local si MLflow absent)
make serve     # API sur http://localhost:8000
```

Exemple d'appel (12 features attendues : feature_0 … feature_11) :

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_0":0.1,"feature_1":-1.2,"feature_2":0.5,"feature_3":0.0,
       "feature_4":0.3,"feature_5":-0.7,"feature_6":1.1,"feature_7":0.2,
       "feature_8":-0.4,"feature_9":0.9,"feature_10":0.0,"feature_11":-0.1}}'
```

## Dashboard Grafana

Un tableau de bord **« Churn API — Service Monitoring »** est pré-provisionné
(débit, latence p50/p95/p99, taux d'erreur, débit par endpoint). Il apparaît
automatiquement sur http://localhost:3000 dès le démarrage, sans configuration.

## Promotion de modèle (champion / challenger)

Le registry utilise un alias **`production`** servi par l'API. La promotion
compare la dernière version au champion en place et ne promeut que si elle est
au moins aussi bonne :

```bash
make promote   # évalue challenger vs production, promeut si meilleur
```
Tant qu'aucun modèle n'est promu, l'API sert automatiquement la dernière version.

## Ré-entraînement automatique sur dérive (Prefect)

Un flux Prefect enchaîne *générer → détecter la dérive → si dérive : ré-entraîner
+ promouvoir* :

```bash
make retrain   # ne ré-entraîne que si une dérive significative est détectée
```

## Démonstration du monitoring de drift

```bash
# 1) baseline : pas de dérive
make data && make train && make promote && make evaluate

# 2) injecter une dérive : mettre drift_shift: 2.0 dans params.yaml, puis
make data && make evaluate   # l'accuracy chute sur le courant drifté
make drift                   # rapport Evidently dans reports/ (dérive détectée)
make retrain                 # le flux ré-entraîne et promeut automatiquement
```

## Commandes utiles

```bash
make help      # liste toutes les cibles
make test      # tests pytest
make lint      # ruff
make down      # arrête la stack
```

## Correspondance `make` → Windows (`run.ps1`)

| `make …` | `.\run.ps1 …` |
|---|---|
| `make up` / `make down` | `.\run.ps1 up` / `.\run.ps1 down` |
| `make data` / `make train` | `.\run.ps1 data` / `.\run.ps1 train` |
| `make evaluate` / `make drift` | `.\run.ps1 evaluate` / `.\run.ps1 drift` |
| `make promote` / `make retrain` | `.\run.ps1 promote` / `.\run.ps1 retrain` |
| `make test` | `.\run.ps1 test` |
| (enchaînement complet) | `.\run.ps1 pipeline` |

## Structure

Voir [`ARCHITECTURE.md`](./ARCHITECTURE.md) pour le détail. En bref :
`src/data` (données), `src/pipelines` (entraînement/éval/promotion/orchestration),
`src/api` (serving), `src/db` (logs de prédictions), `src/monitoring` (drift),
`tests/` (qualité).

## Limites assumées (c'est un squelette pédagogique)

- Modèle volontairement simple ; pas de feature engineering poussé.
- Pas d'authentification sur l'API (à ajouter pour un usage réel).
- Flux Prefect exécuté à la demande ; en production on le planifierait
  (deployment Prefect) ou on le déclencherait sur alerte de monitoring.
