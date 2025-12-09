"""
Script d'initialisation de la base de données PostgreSQL pour le monitoring.

Crée les 4 tables nécessaires au stockage des données de production :
- predictions : Stocke chaque prédiction de l'API
- feature_values : Valeurs des features importantes pour drift detection
- anomalies : Anomalies et alertes détectées
- drift_reports : Métadonnées des rapports Evidently générés

Usage:
    python src/scripts/init_database.py

Requirements:
    - PostgreSQL 16 installé et démarré
    - Base de données 'credit_scoring_prod' créée
    - Utilisateur 'moon' avec les permissions appropriées
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
from pathlib import Path

# Configuration de la base de données
DB_CONFIG = {
    'dbname': 'credit_scoring_prod',
    'user': 'moon',
    'password': 'moon',
    'host': 'localhost',
    'port': '5432'
}

# Seuil métier du modèle (mis à jour après correction SK_ID_CURR)
MODEL_THRESHOLD = 0.5225


def create_tables():
    """Crée les 4 tables du schéma de monitoring."""

    # SQL pour la table predictions
    create_predictions_table = """
    CREATE TABLE IF NOT EXISTS predictions (
        -- Identifiants
        request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

        -- Métadonnées API
        endpoint VARCHAR(50) NOT NULL,
        model_version VARCHAR(20) NOT NULL,
        api_version VARCHAR(20) NOT NULL,

        -- Performance (en millisecondes)
        preprocessing_time_ms FLOAT,
        inference_time_ms FLOAT NOT NULL,
        total_response_time_ms FLOAT NOT NULL,
        memory_usage_mb FLOAT,

        -- Outputs du modèle
        prediction_proba FLOAT NOT NULL,
        prediction_class INT NOT NULL,
        decision VARCHAR(20) NOT NULL,
        confidence_level VARCHAR(10),

        -- Qualité & Erreurs
        http_status_code INT NOT NULL,
        error_message TEXT,
        data_quality_score FLOAT,
        warning_flags TEXT[],

        -- Métadonnées business (anonymisées)
        client_id VARCHAR(50),
        loan_amount FLOAT,
        decision_threshold FLOAT DEFAULT 0.5225,

        CONSTRAINT chk_proba CHECK (prediction_proba BETWEEN 0 AND 1),
        CONSTRAINT chk_class CHECK (prediction_class IN (0, 1))
    );

    CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_decision ON predictions(decision);
    CREATE INDEX IF NOT EXISTS idx_status_code ON predictions(http_status_code);
    """

    # SQL pour la table feature_values
    create_feature_values_table = """
    CREATE TABLE IF NOT EXISTS feature_values (
        id SERIAL PRIMARY KEY,
        request_id UUID NOT NULL REFERENCES predictions(request_id) ON DELETE CASCADE,
        feature_name VARCHAR(100) NOT NULL,
        feature_value FLOAT,
        is_null BOOLEAN DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_feature_name_time ON feature_values(feature_name, request_id);
    """

    # SQL pour la table anomalies
    create_anomalies_table = """
    CREATE TABLE IF NOT EXISTS anomalies (
        id SERIAL PRIMARY KEY,
        request_id UUID REFERENCES predictions(request_id),
        detected_at TIMESTAMP NOT NULL DEFAULT NOW(),

        anomaly_type VARCHAR(50) NOT NULL,
        severity VARCHAR(20) NOT NULL,

        description TEXT NOT NULL,
        affected_features TEXT[],
        metric_value FLOAT,
        threshold_value FLOAT,

        is_resolved BOOLEAN DEFAULT FALSE,
        resolved_at TIMESTAMP,
        resolution_notes TEXT,

        CONSTRAINT chk_severity CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'))
    );

    CREATE INDEX IF NOT EXISTS idx_anomaly_type ON anomalies(anomaly_type);
    CREATE INDEX IF NOT EXISTS idx_severity ON anomalies(severity);
    CREATE INDEX IF NOT EXISTS idx_detected_at ON anomalies(detected_at DESC);
    """

    # SQL pour la table drift_reports
    create_drift_reports_table = """
    CREATE TABLE IF NOT EXISTS drift_reports (
        id SERIAL PRIMARY KEY,
        generated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        report_period_start TIMESTAMP NOT NULL,
        report_period_end TIMESTAMP NOT NULL,

        reference_dataset VARCHAR(100),
        current_dataset_size INT NOT NULL,
        reference_dataset_size INT NOT NULL,

        drift_detected BOOLEAN NOT NULL,
        drift_score FLOAT,
        n_features_drifted INT,
        drifted_features TEXT[],

        report_html_path TEXT NOT NULL,
        report_json_path TEXT,

        metadata JSONB
    );

    CREATE INDEX IF NOT EXISTS idx_report_generated_at ON drift_reports(generated_at DESC);
    """

    try:
        # Connexion à la base de données
        print("Connexion à PostgreSQL...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        print(f"[OK] Connecté à la base de données '{DB_CONFIG['dbname']}'")

        # Création des tables
        print("\nCréation des tables...")

        print("  -> Table 'predictions'...")
        cursor.execute(create_predictions_table)
        print("     [OK] Table 'predictions' créée (avec index)")

        print("  -> Table 'feature_values'...")
        cursor.execute(create_feature_values_table)
        print("     [OK] Table 'feature_values' créée (avec index)")

        print("  -> Table 'anomalies'...")
        cursor.execute(create_anomalies_table)
        print("     [OK] Table 'anomalies' créée (avec index)")

        print("  -> Table 'drift_reports'...")
        cursor.execute(create_drift_reports_table)
        print("     [OK] Table 'drift_reports' créée (avec index)")

        # Vérification
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()

        print(f"\n[SUCCESS] Schéma créé avec succès !")
        print(f"          Tables créées : {', '.join([t[0] for t in tables])}")
        print(f"          Seuil métier configuré : {MODEL_THRESHOLD}")

        # Fermeture
        cursor.close()
        conn.close()

        print("\n[SUCCESS] Initialisation terminée avec succès !")
        return True

    except psycopg2.OperationalError as e:
        print(f"\n[ERROR] Erreur de connexion à PostgreSQL:")
        print(f"        {e}")
        print(f"\nVérifiez que:")
        print(f"  - PostgreSQL est démarré: brew services list")
        print(f"  - La base '{DB_CONFIG['dbname']}' existe: psql -l")
        print(f"  - L'utilisateur '{DB_CONFIG['user']}' a les permissions")
        return False

    except psycopg2.Error as e:
        print(f"\n[ERROR] Erreur PostgreSQL:")
        print(f"        {e}")
        return False

    except Exception as e:
        print(f"\n[ERROR] Erreur inattendue:")
        print(f"        {e}")
        return False


def verify_schema():
    """Vérifie que le schéma a été créé correctement."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("\nVérification du schéma...")

        # Compter les tables
        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """)
        table_count = cursor.fetchone()[0]
        print(f"  [OK] {table_count} tables trouvées")

        # Compter les index
        cursor.execute("""
            SELECT COUNT(*)
            FROM pg_indexes
            WHERE schemaname = 'public';
        """)
        index_count = cursor.fetchone()[0]
        print(f"  [OK] {index_count} index trouvés")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"  [ERROR] Erreur lors de la vérification: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Initialisation de la base de données PostgreSQL")
    print("Projet: Home Credit Scoring API - Monitoring")
    print("=" * 60)

    # Créer les tables
    success = create_tables()

    if success:
        # Vérifier le schéma
        verify_schema()

        print("\n" + "=" * 60)
        print("[SUCCESS] Base de données prête pour le monitoring !")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("[ERROR] Échec de l'initialisation")
        print("=" * 60)
        sys.exit(1)
