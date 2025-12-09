"""
Configuration de l'API.

Centralise tous les paramètres de configuration (chemins, seuils, etc.)
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration de l'application."""

    # Informations de l'API
    api_title: str = "Home Credit Scoring API"
    api_version: str = "1.0.0"
    api_description: str = "API de scoring crédit pour prédire le risque de défaut de paiement"

    # Chemins vers les artefacts du modèle
    models_dir: Path = Path("models")
    model_path: Path = models_dir / "model.pkl"
    label_encoders_path: Path = models_dir / "label_encoders.pkl"
    onehot_encoder_path: Path = models_dir / "onehot_encoder.pkl"
    feature_names_path: Path = models_dir / "feature_names.pkl"
    metrics_path: Path = models_dir / "metrics.json"
    threshold_path: Path = models_dir / "threshold.json"

    # Configuration du serveur
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json ou text
    environment: str = "production"  # "local" ou "production"

    # Limites
    max_batch_size: int = 100
    max_request_timeout: int = 30  # secondes

    # ===== CONFIGURATION POSTGRESQL =====
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "credit_scoring_prod"
    db_user: str = "moon"
    db_password: str = "moon"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_store_predictions: bool = True  # Active/désactive le stockage PostgreSQL

    @property
    def database_url(self) -> str:
        """Construit l'URL de connexion PostgreSQL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instance globale de configuration
settings = Settings()
