"""
Script pour calculer les métriques baseline depuis les données de production PostgreSQL.

Ce script extrait les statistiques détaillées (mean, median, percentiles, throughput)
depuis les 1,166 prédictions stockées dans PostgreSQL pour établir une baseline
de performance avant optimisation.

Usage:
    python src/scripts/calculate_baseline_metrics.py

Output:
    reports/benchmarks/baseline_production.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor


def get_database_url() -> str:
    """Retourne l'URL de connexion PostgreSQL."""
    return "postgresql://moon:moon@localhost:5432/credit_scoring_prod"


def calculate_metrics_from_postgres() -> Dict[str, Any]:
    """
    Calcule les métriques de performance depuis PostgreSQL.

    Returns:
        Dictionnaire avec toutes les métriques (mean, median, percentiles, throughput).
    """
    db_url = get_database_url()

    # Parser l'URL PostgreSQL
    # Format: postgresql://user:password@host:port/database
    parts = db_url.replace("postgresql://", "").split("@")
    user_pass = parts[0].split(":")
    host_port_db = parts[1].split("/")
    host_port = host_port_db[0].split(":")

    conn_params = {
        "user": user_pass[0],
        "password": user_pass[1],
        "host": host_port[0],
        "port": host_port[1],
        "database": host_port_db[1]
    }

    print(f"📊 Connexion à PostgreSQL : {conn_params['database']}@{conn_params['host']}...")

    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Requête SQL pour calculer toutes les statistiques
    query = """
    SELECT
        COUNT(*) as n_samples,
        MIN(timestamp) as date_start,
        MAX(timestamp) as date_end,

        -- Total response time
        ROUND(AVG(total_response_time_ms)::numeric, 2) as total_mean,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_response_time_ms)::numeric, 2) as total_median,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_response_time_ms)::numeric, 2) as total_p95,
        ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_response_time_ms)::numeric, 2) as total_p99,
        ROUND(MIN(total_response_time_ms)::numeric, 2) as total_min,
        ROUND(MAX(total_response_time_ms)::numeric, 2) as total_max,
        ROUND(STDDEV(total_response_time_ms)::numeric, 2) as total_std,

        -- Preprocessing time
        ROUND(AVG(preprocessing_time_ms)::numeric, 2) as prep_mean,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY preprocessing_time_ms)::numeric, 2) as prep_median,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY preprocessing_time_ms)::numeric, 2) as prep_p95,
        ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY preprocessing_time_ms)::numeric, 2) as prep_p99,
        ROUND(MIN(preprocessing_time_ms)::numeric, 2) as prep_min,
        ROUND(MAX(preprocessing_time_ms)::numeric, 2) as prep_max,
        ROUND(STDDEV(preprocessing_time_ms)::numeric, 2) as prep_std,

        -- Inference time
        ROUND(AVG(inference_time_ms)::numeric, 2) as inf_mean,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY inference_time_ms)::numeric, 2) as inf_median,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY inference_time_ms)::numeric, 2) as inf_p95,
        ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY inference_time_ms)::numeric, 2) as inf_p99,
        ROUND(MIN(inference_time_ms)::numeric, 2) as inf_min,
        ROUND(MAX(inference_time_ms)::numeric, 2) as inf_max,
        ROUND(STDDEV(inference_time_ms)::numeric, 2) as inf_std

    FROM predictions
    WHERE
        total_response_time_ms IS NOT NULL
        AND preprocessing_time_ms IS NOT NULL
        AND inference_time_ms IS NOT NULL;
    """

    print("🔍 Exécution de la requête SQL...")
    cursor.execute(query)
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    # Convertir le résultat en dictionnaire Python standard
    result = dict(result)

    # Calculer les pourcentages et le throughput
    prep_percentage = round((float(result['prep_mean']) / float(result['total_mean'])) * 100, 1)
    inf_percentage = round((float(result['inf_mean']) / float(result['total_mean'])) * 100, 1)

    # Throughput = 1000ms / mean_total_ms pour avoir prédictions/seconde
    throughput_per_second = round(1000 / float(result['total_mean']), 2)
    throughput_per_minute = round(throughput_per_second * 60, 0)

    # Construire le JSON de sortie
    metrics = {
        "source": "production_data",
        "database": "credit_scoring_prod",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": result['n_samples'],
        "date_range": {
            "start": result['date_start'].strftime("%Y-%m-%d") if result['date_start'] else None,
            "end": result['date_end'].strftime("%Y-%m-%d") if result['date_end'] else None
        },

        "total_time_ms": {
            "mean": float(result['total_mean']),
            "median": float(result['total_median']),
            "p50": float(result['total_median']),
            "p95": float(result['total_p95']),
            "p99": float(result['total_p99']),
            "min": float(result['total_min']),
            "max": float(result['total_max']),
            "std": float(result['total_std'])
        },

        "preprocessing_time_ms": {
            "mean": float(result['prep_mean']),
            "median": float(result['prep_median']),
            "p50": float(result['prep_median']),
            "p95": float(result['prep_p95']),
            "p99": float(result['prep_p99']),
            "min": float(result['prep_min']),
            "max": float(result['prep_max']),
            "std": float(result['prep_std']),
            "percentage_of_total": prep_percentage
        },

        "inference_time_ms": {
            "mean": float(result['inf_mean']),
            "median": float(result['inf_median']),
            "p50": float(result['inf_median']),
            "p95": float(result['inf_p95']),
            "p99": float(result['inf_p99']),
            "min": float(result['inf_min']),
            "max": float(result['inf_max']),
            "std": float(result['inf_std']),
            "percentage_of_total": inf_percentage
        },

        "throughput": {
            "predictions_per_second": throughput_per_second,
            "predictions_per_minute": int(throughput_per_minute)
        }
    }

    return metrics


def save_metrics_to_json(metrics: Dict[str, Any], output_path: Path) -> None:
    """
    Sauvegarde les métriques dans un fichier JSON.

    Args:
        metrics: Dictionnaire des métriques.
        output_path: Chemin du fichier de sortie.
    """
    # Créer le répertoire si nécessaire
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Métriques sauvegardées : {output_path}")


def print_summary(metrics: Dict[str, Any]) -> None:
    """Affiche un résumé des métriques calculées."""
    print("\n" + "="*70)
    print("📊 MÉTRIQUES BASELINE PRODUCTION")
    print("="*70)
    print(f"\n🗄️  Source : {metrics['source']}")
    print(f"📅 Période : {metrics['date_range']['start']} → {metrics['date_range']['end']}")
    print(f"📈 Échantillons : {metrics['n_samples']:,} prédictions")

    print("\n⏱️  TEMPS TOTAL")
    print(f"   Mean   : {metrics['total_time_ms']['mean']:.2f} ms")
    print(f"   Median : {metrics['total_time_ms']['median']:.2f} ms")
    print(f"   P95    : {metrics['total_time_ms']['p95']:.2f} ms")
    print(f"   P99    : {metrics['total_time_ms']['p99']:.2f} ms")
    print(f"   Min    : {metrics['total_time_ms']['min']:.2f} ms")
    print(f"   Max    : {metrics['total_time_ms']['max']:.2f} ms")
    print(f"   Std    : {metrics['total_time_ms']['std']:.2f} ms")

    print("\n🔧 PREPROCESSING")
    print(f"   Mean   : {metrics['preprocessing_time_ms']['mean']:.2f} ms ({metrics['preprocessing_time_ms']['percentage_of_total']:.1f}%)")
    print(f"   Median : {metrics['preprocessing_time_ms']['median']:.2f} ms")
    print(f"   P95    : {metrics['preprocessing_time_ms']['p95']:.2f} ms")
    print(f"   P99    : {metrics['preprocessing_time_ms']['p99']:.2f} ms")

    print("\n🤖 INFERENCE (LightGBM)")
    print(f"   Mean   : {metrics['inference_time_ms']['mean']:.2f} ms ({metrics['inference_time_ms']['percentage_of_total']:.1f}%)")
    print(f"   Median : {metrics['inference_time_ms']['median']:.2f} ms")
    print(f"   P95    : {metrics['inference_time_ms']['p95']:.2f} ms")
    print(f"   P99    : {metrics['inference_time_ms']['p99']:.2f} ms")

    print("\n🚀 THROUGHPUT")
    print(f"   {metrics['throughput']['predictions_per_second']:.2f} prédictions/seconde")
    print(f"   {metrics['throughput']['predictions_per_minute']:,} prédictions/minute")

    print("\n🔴 GOULOT D'ÉTRANGLEMENT IDENTIFIÉ")
    print(f"   Preprocessing : {metrics['preprocessing_time_ms']['percentage_of_total']:.1f}% du temps total")
    print(f"   → Focus optimisation sur ce composant")

    print("\n" + "="*70)


def main():
    """Fonction principale."""
    print("\n🚀 Calcul des métriques baseline depuis PostgreSQL\n")

    try:
        # Calculer les métriques
        metrics = calculate_metrics_from_postgres()

        # Afficher le résumé
        print_summary(metrics)

        # Sauvegarder le JSON
        output_path = Path("reports/benchmarks/baseline_production.json")
        save_metrics_to_json(metrics, output_path)

        print(f"\n✅ Tâche 1.1 terminée : Métriques baseline PostgreSQL calculées")
        print(f"📁 Fichier généré : {output_path}")

        return 0

    except Exception as e:
        print(f"\n❌ Erreur : {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
