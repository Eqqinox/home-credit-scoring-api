"""
Module de détection de data drift avec Evidently AI.

Détecte les dérives de données entre dataset de référence et production.
Génère des rapports HTML et JSON pour analyse et visualisation.
"""
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Détecteur de data drift utilisant Evidently AI.

    Compare les données de production avec un dataset de référence
    pour identifier les changements de distribution des features.

    Attributes:
        reference_data (pd.DataFrame): Dataset de référence (train)
        drift_threshold (float): Seuil d'alerte (% features en drift)
    """

    def __init__(
        self,
        reference_data_path: str,
        drift_threshold: float = 0.3,
        target_col: Optional[str] = None
    ):
        """
        Initialise le détecteur de drift.

        Args:
            reference_data_path: Chemin vers le dataset de référence (Parquet ou CSV)
            drift_threshold: Seuil d'alerte (0.3 = 30% features en drift)
            target_col: Nom de la colonne cible (None si pas de target)

        Raises:
            FileNotFoundError: Si le fichier de référence n'existe pas
            ValueError: Si le dataset est vide
        """
        self.reference_data_path = Path(reference_data_path)
        self.drift_threshold = drift_threshold
        self.target_col = target_col

        # Charger les données de référence
        self.reference_data = self._load_reference_data()

        logger.info(
            f"DriftDetector initialisé - Référence: {len(self.reference_data)} lignes, "
            f"Seuil: {drift_threshold*100:.0f}%"
        )

    def _load_reference_data(self) -> pd.DataFrame:
        """
        Charge le dataset de référence.

        Returns:
            DataFrame de référence

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le dataset est vide
        """
        if not self.reference_data_path.exists():
            raise FileNotFoundError(f"Fichier de référence introuvable: {self.reference_data_path}")

        # Charger selon l'extension
        if self.reference_data_path.suffix == '.parquet':
            df = pd.read_parquet(self.reference_data_path)
        elif self.reference_data_path.suffix == '.csv':
            df = pd.read_csv(self.reference_data_path)
        else:
            raise ValueError(f"Format non supporté: {self.reference_data_path.suffix}")

        if df.empty:
            raise ValueError("Le dataset de référence est vide")

        logger.info(f"Dataset de référence chargé: {df.shape}")
        return df

    def generate_drift_report(
        self,
        production_data: pd.DataFrame,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """
        Génère un rapport de drift entre référence et production.

        Args:
            production_data: DataFrame des données de production
            period_start: Date de début de la période analysée
            period_end: Date de fin de la période analysée

        Returns:
            Dictionnaire contenant :
                - drift_detected (bool): Drift détecté ou non
                - drift_score (float): Score global de drift (0-1)
                - n_features_drifted (int): Nombre de features en drift
                - drifted_features (List[str]): Liste des features affectées
                - report (Report): Objet Evidently Report
                - metadata (Dict): Métadonnées (tailles datasets, période, etc.)

        Raises:
            ValueError: Si production_data est vide ou incompatible
        """
        if production_data.empty:
            raise ValueError("Les données de production sont vides")

        # Vérifier que les colonnes sont compatibles
        ref_cols = set(self.reference_data.columns)
        prod_cols = set(production_data.columns)

        # Utiliser les colonnes communes
        common_cols = list(ref_cols.intersection(prod_cols))

        if not common_cols:
            raise ValueError("Aucune colonne commune entre référence et production")

        logger.info(f"Analyse drift sur {len(common_cols)} features communes")

        # Filtrer les colonnes communes
        ref_subset = self.reference_data[common_cols].copy()
        prod_subset = production_data[common_cols].copy()

        # Filtrer les colonnes vides (toutes les valeurs NULL)
        # Evidently ne peut pas analyser les colonnes vides
        non_empty_cols = []
        for col in common_cols:
            if not prod_subset[col].isna().all() and not ref_subset[col].isna().all():
                non_empty_cols.append(col)

        if not non_empty_cols:
            raise ValueError("Toutes les colonnes sont vides, impossible d'analyser le drift")

        logger.info(f"Filtrage: {len(common_cols) - len(non_empty_cols)} colonnes vides retirées")
        logger.info(f"Analyse sur {len(non_empty_cols)} colonnes non-vides")

        ref_subset = ref_subset[non_empty_cols]
        prod_subset = prod_subset[non_empty_cols]

        # Créer le rapport Evidently
        report = Report(metrics=[DataDriftPreset()])

        try:
            # IMPORTANT: run() retourne un Snapshot, pas void
            snapshot = report.run(
                reference_data=ref_subset,
                current_data=prod_subset
            )
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport Evidently: {e}")
            raise

        # Extraire les résultats du snapshot
        report_dict = snapshot.dict()

        # Parcourir les métriques pour trouver le dataset drift
        drift_metrics = report_dict.get('metrics', [])
        dataset_drift_metric = None

        for metric in drift_metrics:
            if metric.get('metric') == 'DatasetDriftMetric':
                dataset_drift_metric = metric.get('result', {})
                break

        if not dataset_drift_metric:
            logger.warning("DatasetDriftMetric non trouvé dans le rapport")
            drift_detected = False
            drift_score = 0.0
            n_drifted = 0
            drifted_features = []
        else:
            # Extraire les informations de drift
            drift_detected = dataset_drift_metric.get('dataset_drift', False)
            drift_score = dataset_drift_metric.get('drift_share', 0.0)
            n_drifted = dataset_drift_metric.get('number_of_drifted_columns', 0)

            # Liste des features en drift
            drift_by_columns = dataset_drift_metric.get('drift_by_columns', {})
            drifted_features = [
                col for col, metrics in drift_by_columns.items()
                if metrics.get('drift_detected', False)
            ]

        # Métadonnées
        metadata = {
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'reference_dataset_size': len(ref_subset),
            'current_dataset_size': len(prod_subset),
            'n_features_analyzed': len(common_cols),
            'drift_threshold': self.drift_threshold,
            'generated_at': datetime.now().isoformat()
        }

        logger.info(
            f"Rapport généré - Drift: {'OUI' if drift_detected else 'NON'}, "
            f"Score: {drift_score:.2f}, Features affectées: {n_drifted}/{len(common_cols)}"
        )

        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'n_features_drifted': n_drifted,
            'drifted_features': drifted_features,
            'snapshot': snapshot,  # Retourne le Snapshot, pas le Report
            'metadata': metadata
        }

    def save_report_to_file(
        self,
        snapshot,  # Type: Snapshot (import depuis evidently.core.report)
        output_dir: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Sauvegarde le rapport Evidently en HTML et JSON.

        Args:
            snapshot: Objet Evidently Snapshot (retourné par report.run())
            output_dir: Dossier de destination
            filename: Nom du fichier (sans extension), si None utilise timestamp

        Returns:
            Chemin du fichier HTML généré

        Raises:
            IOError: Si l'écriture échoue
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Générer nom de fichier avec timestamp si non fourni
        if filename is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"drift_report_{timestamp}"

        # Sauvegarder HTML
        html_file = output_path / f"{filename}.html"
        try:
            snapshot.save_html(str(html_file))
            logger.info(f"Rapport HTML sauvegardé: {html_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde HTML: {e}")
            raise IOError(f"Impossible de sauvegarder le rapport HTML: {e}")

        # Sauvegarder JSON
        json_file = output_path / f"{filename}.json"
        try:
            snapshot.save_json(str(json_file))
            logger.info(f"Rapport JSON sauvegardé: {json_file}")
        except Exception as e:
            logger.warning(f"Erreur lors de la sauvegarde JSON: {e}")

        return str(html_file)

    def get_drifted_features(self, report_result: Dict[str, Any]) -> List[str]:
        """
        Extrait la liste des features en drift.

        Args:
            report_result: Résultat de generate_drift_report()

        Returns:
            Liste des noms de features en drift
        """
        return report_result.get('drifted_features', [])

    def check_alert_threshold(self, report_result: Dict[str, Any]) -> bool:
        """
        Vérifie si le drift dépasse le seuil d'alerte.

        Args:
            report_result: Résultat de generate_drift_report()

        Returns:
            True si alerte (drift_score > drift_threshold)
        """
        drift_score = report_result.get('drift_score', 0.0)
        alert = drift_score > self.drift_threshold

        if alert:
            logger.warning(
                f"⚠️ ALERTE DRIFT - Score: {drift_score:.2%} > Seuil: {self.drift_threshold:.2%}"
            )

        return alert
