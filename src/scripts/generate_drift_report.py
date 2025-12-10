"""
Script de génération de rapports de data drift avec Evidently AI.

Génère un rapport comparant les données de production récentes
avec le dataset de référence (train). Sauvegarde le rapport en HTML
et enregistre les métadonnées en PostgreSQL.

Usage:
    python src/scripts/generate_drift_report.py
    python src/scripts/generate_drift_report.py --days 7 --threshold 0.3
    python src/scripts/generate_drift_report.py --output-dir reports/drift
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.monitoring.drift_detector import DriftDetector
from src.monitoring.storage import PredictionStorage
from src.api.config import settings

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Génère un rapport de data drift avec Evidently AI"
    )

    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Nombre de jours de données de production à analyser (défaut: 7)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Seuil d\'alerte pour le drift (0-1, défaut: 0.3 = 30%%)'
    )

    parser.add_argument(
        '--reference-data',
        type=str,
        default='data/reference/train_reference.parquet',
        help='Chemin vers le dataset de référence (défaut: data/reference/train_reference.parquet)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/drift',
        help='Dossier de sortie pour les rapports HTML (défaut: reports/drift)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='Nombre max de prédictions à analyser (défaut: 1000)'
    )

    return parser.parse_args()


def main():
    """
    Workflow principal de génération de rapport.

    Steps:
        1. Charger dataset de référence
        2. Récupérer données de production
        3. Générer rapport Evidently
        4. Sauvegarder HTML
        5. Enregistrer en PostgreSQL
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("GÉNÉRATION RAPPORT DATA DRIFT")
    logger.info("=" * 60)
    logger.info(f"Période: {args.days} derniers jours")
    logger.info(f"Seuil d'alerte: {args.threshold * 100:.0f}%")
    logger.info(f"Référence: {args.reference_data}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    try:
        # ===== 1. Initialiser DriftDetector =====
        logger.info("📂 Chargement dataset de référence...")

        reference_path = Path(args.reference_data)
        if not reference_path.exists():
            logger.error(f"❌ Dataset de référence introuvable: {reference_path}")
            sys.exit(1)

        detector = DriftDetector(
            reference_data_path=str(reference_path),
            drift_threshold=args.threshold
        )

        logger.info(f"✅ Dataset de référence chargé: {len(detector.reference_data)} lignes")

        # ===== 2. Récupérer données de production =====
        logger.info(f"📊 Récupération données de production ({args.days} jours)...")

        storage = PredictionStorage(
            database_url=settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow
        )

        production_data = storage.get_production_features(
            days=args.days,
            limit=args.limit
        )

        if production_data.empty:
            logger.error(f"❌ Aucune donnée de production trouvée (derniers {args.days} jours)")
            logger.info("💡 Générez du trafic avec: python src/scripts/simulate_traffic.py")
            storage.close()
            sys.exit(1)

        logger.info(f"✅ Données de production récupérées: {len(production_data)} lignes")

        # ===== 3. Générer rapport Evidently =====
        logger.info("⚙️ Génération rapport Evidently AI...")

        period_end = datetime.now()
        period_start = period_end - timedelta(days=args.days)

        # Filtrer le dataset de référence pour avoir les mêmes colonnes
        common_cols = list(set(detector.reference_data.columns).intersection(set(production_data.columns)))

        if not common_cols:
            logger.error("❌ Aucune colonne commune entre référence et production")
            storage.close()
            sys.exit(1)

        logger.info(f"📋 Analyse sur {len(common_cols)} features communes")

        report_result = detector.generate_drift_report(
            production_data=production_data,
            period_start=period_start,
            period_end=period_end
        )

        logger.info(f"✅ Rapport généré")
        logger.info(f"   - Drift détecté: {'⚠️ OUI' if report_result['drift_detected'] else '✅ NON'}")
        logger.info(f"   - Score de drift: {report_result['drift_score']:.2%}")
        logger.info(f"   - Features affectées: {report_result['n_features_drifted']}/{len(common_cols)}")

        if report_result['drifted_features']:
            logger.info(f"   - Features en drift: {', '.join(report_result['drifted_features'][:5])}")
            if len(report_result['drifted_features']) > 5:
                logger.info(f"     ... et {len(report_result['drifted_features']) - 5} autres")

        # Vérifier seuil d'alerte
        if detector.check_alert_threshold(report_result):
            logger.warning(f"⚠️ ALERTE: Score de drift ({report_result['drift_score']:.2%}) > seuil ({args.threshold:.2%})")

        # ===== 4. Sauvegarder rapport HTML =====
        logger.info(f"💾 Sauvegarde rapport HTML...")

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"drift_report_{timestamp}"

        html_path = detector.save_report_to_file(
            snapshot=report_result['snapshot'],
            output_dir=args.output_dir,
            filename=filename
        )

        logger.info(f"✅ Rapport HTML sauvegardé: {html_path}")

        # ===== 5. Enregistrer en PostgreSQL =====
        logger.info("🗄️ Enregistrement en PostgreSQL...")

        json_path = html_path.replace('.html', '.json')

        success = storage.save_drift_report(
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            reference_dataset=str(reference_path.name),
            current_dataset_size=len(production_data),
            reference_dataset_size=len(detector.reference_data),
            drift_detected=report_result['drift_detected'],
            drift_score=report_result['drift_score'],
            n_features_drifted=report_result['n_features_drifted'],
            drifted_features=report_result['drifted_features'],
            report_html_path=html_path,
            report_json_path=json_path if Path(json_path).exists() else None,
            metadata=report_result['metadata']
        )

        if success:
            logger.info("✅ Rapport enregistré en PostgreSQL")
        else:
            logger.warning("⚠️ Échec de l'enregistrement en PostgreSQL")

        storage.close()

        # ===== 6. Résumé =====
        logger.info("=" * 60)
        logger.info("✅ RAPPORT GÉNÉRÉ AVEC SUCCÈS")
        logger.info("=" * 60)
        logger.info(f"📄 Rapport HTML: {html_path}")
        logger.info(f"🗄️ PostgreSQL: {'✅ Enregistré' if success else '❌ Échec'}")
        logger.info(f"📊 Drift: {'⚠️ DÉTECTÉ' if report_result['drift_detected'] else '✅ NON DÉTECTÉ'}")
        logger.info("=" * 60)

        # Ouvrir le rapport dans le navigateur (optionnel)
        import webbrowser
        logger.info(f"🌐 Ouverture du rapport dans le navigateur...")
        webbrowser.open(f"file://{Path(html_path).resolve()}")

    except KeyboardInterrupt:
        logger.info("\n⚠️ Interruption utilisateur")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération du rapport: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
