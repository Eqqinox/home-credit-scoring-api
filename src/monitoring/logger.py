"""
Logging structuré JSON pour monitoring production.
Utilise structlog pour format JSON avec contexte riche.
"""

import structlog
import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime


def configure_structlog(log_level: str = "INFO", environment: str = "production"):
    """
    Configure structlog pour output JSON.

    Args:
        log_level: DEBUG (local) ou INFO (production)
        environment: "local" ou "production"
    """

    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if environment == "local":
        # Console couleur pour dev
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # JSON pur pour production
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure logging standard Python aussi
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


class ProductionLogger:
    """
    Logger structuré pour prédictions et événements API.
    """

    def __init__(self):
        self.logger = structlog.get_logger(__name__)

    def log_prediction(
        self,
        client_id: Optional[int],
        probability: float,
        prediction: int,
        decision: str,
        risk_level: str,
        preprocessing_time_ms: float,
        inference_time_ms: float,
        total_time_ms: float,
        threshold: float,
        model_version: str,
        endpoint: str = "/predict",
        http_status: int = 200,
        error_message: Optional[str] = None
    ):
        """Log une prédiction complète au format JSON."""

        self.logger.info(
            "prediction_completed",
            event_type="prediction",
            client_id=client_id,
            probability_default=probability,
            prediction_class=prediction,
            decision=decision,
            risk_level=risk_level,
            timing={
                "preprocessing_ms": preprocessing_time_ms,
                "inference_ms": inference_time_ms,
                "total_ms": total_time_ms,
            },
            model={
                "version": model_version,
                "threshold": threshold,
            },
            request={
                "endpoint": endpoint,
                "http_status": http_status,
            },
            error=error_message,
            timestamp=datetime.utcnow().isoformat(),
        )

    def log_batch(
        self,
        n_clients: int,
        total_time_ms: float,
        n_success: int,
        n_errors: int,
    ):
        """Log une prédiction batch."""

        self.logger.info(
            "batch_prediction_completed",
            event_type="batch",
            n_clients=n_clients,
            n_success=n_success,
            n_errors=n_errors,
            total_time_ms=total_time_ms,
            avg_time_per_client_ms=total_time_ms / n_clients if n_clients > 0 else 0,
            timestamp=datetime.utcnow().isoformat(),
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        endpoint: str,
        client_id: Optional[int] = None,
        stack_trace: Optional[str] = None,
    ):
        """Log une erreur."""

        self.logger.error(
            "api_error",
            event_type="error",
            error_type=error_type,
            error_message=error_message,
            endpoint=endpoint,
            client_id=client_id,
            stack_trace=stack_trace,
            timestamp=datetime.utcnow().isoformat(),
        )
