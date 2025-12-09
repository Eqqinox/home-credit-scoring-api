"""
Tests pour le logging structuré avec structlog.

Teste la configuration et le format de sortie des logs.
"""

import pytest
import structlog
import logging
import json

from src.monitoring.logger import configure_structlog, ProductionLogger


class TestStructlogConfiguration:
    """Tests de configuration de structlog."""

    def test_configure_structlog_local(self):
        """Test configuration en mode local (dev)."""
        configure_structlog(log_level="DEBUG", environment="local")

        # Vérifier que le logger est configuré
        logger = structlog.get_logger()
        assert logger is not None

    def test_configure_structlog_production(self):
        """Test configuration en mode production."""
        configure_structlog(log_level="INFO", environment="production")

        # Vérifier que le logger est configuré
        logger = structlog.get_logger()
        assert logger is not None


class TestProductionLogger:
    """Tests du ProductionLogger."""

    def setup_method(self):
        """Setup avant chaque test."""
        configure_structlog(log_level="INFO", environment="production")
        self.logger = ProductionLogger()

    def test_log_prediction_format(self, caplog):
        """Test le format JSON d'une prédiction."""
        with caplog.at_level(logging.INFO):
            self.logger.log_prediction(
                client_id=123,
                probability=0.35,
                prediction=0,
                decision="APPROVE",
                risk_level="MEDIUM",
                preprocessing_time_ms=5.2,
                inference_time_ms=2.1,
                total_time_ms=7.5,
                threshold=0.5225,
                model_version="1.0.0",
                endpoint="/predict",
                http_status=200,
            )

        # Récupérer le log émis
        assert len(caplog.records) > 0
        log_message = caplog.records[0].message

        # Parser le JSON
        log_entry = json.loads(log_message)

        # Vérifier les champs obligatoires
        assert log_entry["event"] == "prediction_completed"
        assert log_entry["event_type"] == "prediction"
        assert log_entry["client_id"] == 123
        assert log_entry["probability_default"] == 0.35
        assert log_entry["prediction_class"] == 0
        assert log_entry["decision"] == "APPROVE"
        assert log_entry["risk_level"] == "MEDIUM"

        # Vérifier les timings
        assert "timing" in log_entry
        assert log_entry["timing"]["preprocessing_ms"] == 5.2
        assert log_entry["timing"]["inference_ms"] == 2.1
        assert log_entry["timing"]["total_ms"] == 7.5

        # Vérifier les métadonnées du modèle
        assert "model" in log_entry
        assert log_entry["model"]["version"] == "1.0.0"
        assert log_entry["model"]["threshold"] == 0.5225

        # Vérifier les métadonnées de la requête
        assert "request" in log_entry
        assert log_entry["request"]["endpoint"] == "/predict"
        assert log_entry["request"]["http_status"] == 200

        # Vérifier le timestamp
        assert "timestamp" in log_entry

    def test_log_batch_format(self, caplog):
        """Test le format JSON d'un batch."""
        with caplog.at_level(logging.INFO):
            self.logger.log_batch(
                n_clients=10,
                total_time_ms=150.5,
                n_success=9,
                n_errors=1,
            )

        # Récupérer le log émis
        assert len(caplog.records) > 0
        log_message = caplog.records[0].message

        # Parser le JSON
        log_entry = json.loads(log_message)

        # Vérifier les champs
        assert log_entry["event"] == "batch_prediction_completed"
        assert log_entry["event_type"] == "batch"
        assert log_entry["n_clients"] == 10
        assert log_entry["n_success"] == 9
        assert log_entry["n_errors"] == 1
        assert log_entry["total_time_ms"] == 150.5
        assert log_entry["avg_time_per_client_ms"] == 150.5 / 10
        assert "timestamp" in log_entry

    def test_log_error_format(self, caplog):
        """Test le format JSON d'une erreur."""
        with caplog.at_level(logging.ERROR):
            self.logger.log_error(
                error_type="ValidationError",
                error_message="Feature manquante: EXT_SOURCE_1",
                endpoint="/predict",
                client_id=456,
                stack_trace="Traceback...",
            )

        # Récupérer le log émis
        assert len(caplog.records) > 0
        log_message = caplog.records[0].message

        # Parser le JSON
        log_entry = json.loads(log_message)

        # Vérifier les champs
        assert log_entry["event"] == "api_error"
        assert log_entry["event_type"] == "error"
        assert log_entry["error_type"] == "ValidationError"
        assert log_entry["error_message"] == "Feature manquante: EXT_SOURCE_1"
        assert log_entry["endpoint"] == "/predict"
        assert log_entry["client_id"] == 456
        assert log_entry["stack_trace"] == "Traceback..."
        assert "timestamp" in log_entry

    def test_timings_are_positive(self, caplog):
        """Test que les timings sont positifs et cohérents."""
        preprocessing_ms = 5.5
        inference_ms = 2.2
        total_ms = preprocessing_ms + inference_ms + 0.3

        with caplog.at_level(logging.INFO):
            self.logger.log_prediction(
                client_id=789,
                probability=0.75,
                prediction=1,
                decision="REFUSE",
                risk_level="HIGH",
                preprocessing_time_ms=preprocessing_ms,
                inference_time_ms=inference_ms,
                total_time_ms=total_ms,
                threshold=0.5225,
                model_version="1.0.0",
            )

        assert len(caplog.records) > 0
        log_message = caplog.records[0].message
        log_entry = json.loads(log_message)

        # Vérifier que tous les timings sont positifs
        assert log_entry["timing"]["preprocessing_ms"] > 0
        assert log_entry["timing"]["inference_ms"] > 0
        assert log_entry["timing"]["total_ms"] > 0

        # Vérifier la cohérence (total >= preprocessing + inference)
        assert log_entry["timing"]["total_ms"] >= (
            log_entry["timing"]["preprocessing_ms"] +
            log_entry["timing"]["inference_ms"]
        )


class TestLoggingEdgeCases:
    """Tests des cas limites."""

    def setup_method(self):
        """Setup avant chaque test."""
        configure_structlog(log_level="INFO", environment="production")
        self.logger = ProductionLogger()

    def test_log_prediction_without_client_id(self, caplog):
        """Test prédiction sans client_id (anonyme)."""
        with caplog.at_level(logging.INFO):
            self.logger.log_prediction(
                client_id=None,
                probability=0.45,
                prediction=0,
                decision="APPROVE",
                risk_level="MEDIUM",
                preprocessing_time_ms=5.0,
                inference_time_ms=2.0,
                total_time_ms=7.0,
                threshold=0.5225,
                model_version="1.0.0",
            )

        assert len(caplog.records) > 0
        log_message = caplog.records[0].message
        log_entry = json.loads(log_message)

        # Vérifier que client_id est None (pas omis)
        assert log_entry["client_id"] is None

    def test_log_error_without_stack_trace(self, caplog):
        """Test erreur sans stack trace."""
        with caplog.at_level(logging.ERROR):
            self.logger.log_error(
                error_type="NetworkError",
                error_message="Timeout",
                endpoint="/predict-batch",
            )

        assert len(caplog.records) > 0
        log_message = caplog.records[0].message
        log_entry = json.loads(log_message)

        # Vérifier que les champs optionnels sont présents mais None
        assert log_entry["client_id"] is None
        assert log_entry["stack_trace"] is None

    def test_log_batch_with_zero_clients(self, caplog):
        """Test batch vide (edge case)."""
        with caplog.at_level(logging.INFO):
            self.logger.log_batch(
                n_clients=0,
                total_time_ms=0.5,
                n_success=0,
                n_errors=0,
            )

        assert len(caplog.records) > 0
        log_message = caplog.records[0].message
        log_entry = json.loads(log_message)

        # Vérifier la gestion du cas n_clients=0
        assert log_entry["avg_time_per_client_ms"] == 0
