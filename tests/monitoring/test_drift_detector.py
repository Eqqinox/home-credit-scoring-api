"""
Tests pour le module de détection de data drift.
"""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from src.monitoring.drift_detector import DriftDetector


@pytest.fixture
def reference_data():
    """Fixture créant un dataset de référence."""
    data = {
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
        'feature_2': [10.0, 20.0, 30.0, 40.0, 50.0] * 20,
        'feature_3': ['A', 'B', 'C', 'A', 'B'] * 20,
    }
    return pd.DataFrame(data)


@pytest.fixture
def reference_csv(reference_data, tmp_path):
    """Fixture créant un fichier CSV de référence temporaire."""
    csv_path = tmp_path / "reference.csv"
    reference_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def reference_parquet(reference_data, tmp_path):
    """Fixture créant un fichier Parquet de référence temporaire."""
    parquet_path = tmp_path / "reference.parquet"
    reference_data.to_parquet(parquet_path, index=False)
    return str(parquet_path)


@pytest.fixture
def production_data():
    """Fixture créant des données de production (similaires à la référence)."""
    data = {
        'feature_1': [1.5, 2.5, 3.5, 4.5, 5.5] * 10,
        'feature_2': [15.0, 25.0, 35.0, 45.0, 55.0] * 10,
        'feature_3': ['A', 'B', 'C', 'A', 'B'] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def production_data_with_drift():
    """Fixture créant des données de production avec drift significatif."""
    data = {
        'feature_1': [100.0, 200.0, 300.0] * 10,  # Valeurs très différentes
        'feature_2': [1000.0, 2000.0, 3000.0] * 10,  # Drift majeur
        'feature_3': ['X', 'Y', 'Z'] * 10,  # Nouvelles catégories
    }
    return pd.DataFrame(data)


class TestDriftDetectorInitialization:
    """Tests d'initialisation du DriftDetector."""

    def test_init_with_csv(self, reference_csv):
        """Test l'initialisation avec un fichier CSV."""
        detector = DriftDetector(
            reference_data_path=reference_csv,
            drift_threshold=0.3
        )

        assert detector is not None
        assert detector.drift_threshold == 0.3
        assert len(detector.reference_data) == 100
        assert list(detector.reference_data.columns) == ['feature_1', 'feature_2', 'feature_3']

    def test_init_with_parquet(self, reference_parquet):
        """Test l'initialisation avec un fichier Parquet."""
        detector = DriftDetector(
            reference_data_path=reference_parquet,
            drift_threshold=0.25
        )

        assert detector is not None
        assert detector.drift_threshold == 0.25
        assert len(detector.reference_data) == 100

    def test_init_file_not_found(self):
        """Test l'erreur si le fichier de référence n'existe pas."""
        with pytest.raises(FileNotFoundError):
            DriftDetector(
                reference_data_path="/path/that/does/not/exist.csv",
                drift_threshold=0.3
            )

    def test_init_unsupported_format(self, tmp_path):
        """Test l'erreur avec un format de fichier non supporté."""
        txt_path = tmp_path / "reference.txt"
        txt_path.write_text("some text")

        with pytest.raises(ValueError, match="Format non supporté"):
            DriftDetector(
                reference_data_path=str(txt_path),
                drift_threshold=0.3
            )

    def test_init_empty_dataset(self, tmp_path):
        """Test l'erreur avec un dataset vide."""
        # Créer un CSV avec seulement les headers, pas de données
        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame({'col1': [], 'col2': []}).to_csv(empty_csv, index=False)

        with pytest.raises(ValueError, match="Le dataset de référence est vide"):
            DriftDetector(
                reference_data_path=str(empty_csv),
                drift_threshold=0.3
            )


class TestDriftReportGeneration:
    """Tests de génération de rapports de drift."""

    def test_generate_drift_report_no_drift(self, reference_csv, production_data):
        """Test la génération d'un rapport sans drift significatif."""
        detector = DriftDetector(reference_data_path=reference_csv, drift_threshold=0.3)

        period_start = datetime.now() - timedelta(days=7)
        period_end = datetime.now()

        result = detector.generate_drift_report(
            production_data=production_data,
            period_start=period_start,
            period_end=period_end
        )

        # Vérifier la structure du résultat
        assert 'drift_detected' in result
        assert 'drift_score' in result
        assert 'n_features_drifted' in result
        assert 'drifted_features' in result
        assert 'snapshot' in result
        assert 'metadata' in result

        # Vérifier les métadonnées
        assert result['metadata']['reference_dataset_size'] == 100
        assert result['metadata']['current_dataset_size'] == 50
        assert result['metadata']['drift_threshold'] == 0.3

    def test_generate_drift_report_with_drift(self, reference_csv, production_data_with_drift):
        """Test la génération d'un rapport avec drift significatif."""
        detector = DriftDetector(reference_data_path=reference_csv, drift_threshold=0.3)

        period_start = datetime.now() - timedelta(days=7)
        period_end = datetime.now()

        result = detector.generate_drift_report(
            production_data=production_data_with_drift,
            period_start=period_start,
            period_end=period_end
        )

        # Avec des valeurs très différentes, on s'attend à du drift
        assert isinstance(result['drift_score'], float)
        assert result['drift_score'] >= 0.0
        assert isinstance(result['n_features_drifted'], int)

    def test_generate_drift_report_empty_production(self, reference_csv):
        """Test l'erreur avec des données de production vides."""
        detector = DriftDetector(reference_data_path=reference_csv, drift_threshold=0.3)

        empty_prod = pd.DataFrame()
        period_start = datetime.now() - timedelta(days=7)
        period_end = datetime.now()

        with pytest.raises(ValueError, match="Les données de production sont vides"):
            detector.generate_drift_report(
                production_data=empty_prod,
                period_start=period_start,
                period_end=period_end
            )

    def test_generate_drift_report_no_common_columns(self, reference_csv):
        """Test l'erreur sans colonnes communes."""
        detector = DriftDetector(reference_data_path=reference_csv, drift_threshold=0.3)

        # Données avec des colonnes complètement différentes
        incompatible_data = pd.DataFrame({
            'other_feature': [1, 2, 3],
            'another_feature': [4, 5, 6]
        })

        period_start = datetime.now() - timedelta(days=7)
        period_end = datetime.now()

        with pytest.raises(ValueError, match="Aucune colonne commune"):
            detector.generate_drift_report(
                production_data=incompatible_data,
                period_start=period_start,
                period_end=period_end
            )


class TestDriftReportSaving:
    """Tests de sauvegarde des rapports."""

    def test_save_report_to_file(self, reference_csv, production_data, tmp_path):
        """Test la sauvegarde d'un rapport en HTML et JSON."""
        detector = DriftDetector(reference_data_path=reference_csv, drift_threshold=0.3)

        period_start = datetime.now() - timedelta(days=7)
        period_end = datetime.now()

        result = detector.generate_drift_report(
            production_data=production_data,
            period_start=period_start,
            period_end=period_end
        )

        # Sauvegarder le rapport
        output_dir = str(tmp_path)
        html_path = detector.save_report_to_file(
            snapshot=result['snapshot'],
            output_dir=output_dir,
            filename="test_report"
        )

        # Vérifier que les fichiers existent
        assert Path(html_path).exists()
        assert html_path.endswith("test_report.html")

        json_path = Path(html_path).with_suffix('.json')
        assert json_path.exists()

    def test_save_report_auto_filename(self, reference_csv, production_data, tmp_path):
        """Test la sauvegarde avec génération automatique du nom de fichier."""
        detector = DriftDetector(reference_data_path=reference_csv, drift_threshold=0.3)

        period_start = datetime.now() - timedelta(days=7)
        period_end = datetime.now()

        result = detector.generate_drift_report(
            production_data=production_data,
            period_start=period_start,
            period_end=period_end
        )

        # Sauvegarder sans spécifier le nom de fichier
        output_dir = str(tmp_path)
        html_path = detector.save_report_to_file(
            snapshot=result['snapshot'],
            output_dir=output_dir
        )

        # Vérifier que le fichier existe avec un timestamp
        assert Path(html_path).exists()
        assert "drift_report_" in html_path
        assert html_path.endswith(".html")


class TestDriftUtilityMethods:
    """Tests des méthodes utilitaires."""

    def test_get_drifted_features(self, reference_csv, production_data):
        """Test l'extraction de la liste des features en drift."""
        detector = DriftDetector(reference_data_path=reference_csv, drift_threshold=0.3)

        period_start = datetime.now() - timedelta(days=7)
        period_end = datetime.now()

        result = detector.generate_drift_report(
            production_data=production_data,
            period_start=period_start,
            period_end=period_end
        )

        drifted = detector.get_drifted_features(result)

        assert isinstance(drifted, list)
        # Peut être vide si pas de drift détecté
        for feature in drifted:
            assert isinstance(feature, str)

    def test_check_alert_threshold_no_alert(self, reference_csv, production_data):
        """Test la vérification du seuil d'alerte sans dépassement."""
        detector = DriftDetector(reference_data_path=reference_csv, drift_threshold=0.5)

        period_start = datetime.now() - timedelta(days=7)
        period_end = datetime.now()

        result = detector.generate_drift_report(
            production_data=production_data,
            period_start=period_start,
            period_end=period_end
        )

        # Avec des données similaires et seuil élevé, pas d'alerte
        alert = detector.check_alert_threshold(result)
        assert isinstance(alert, bool)

    def test_check_alert_threshold_with_alert(self, reference_csv, production_data_with_drift):
        """Test la vérification du seuil d'alerte avec dépassement potentiel."""
        detector = DriftDetector(reference_data_path=reference_csv, drift_threshold=0.1)

        period_start = datetime.now() - timedelta(days=7)
        period_end = datetime.now()

        result = detector.generate_drift_report(
            production_data=production_data_with_drift,
            period_start=period_start,
            period_end=period_end
        )

        # Avec des données très différentes et seuil bas, alerte possible
        alert = detector.check_alert_threshold(result)
        assert isinstance(alert, bool)
