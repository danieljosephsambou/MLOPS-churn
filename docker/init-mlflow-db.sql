-- Crée la base dédiée au backend store de MLflow (séparée de la base applicative).
-- Exécuté automatiquement au premier démarrage du conteneur Postgres.
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
