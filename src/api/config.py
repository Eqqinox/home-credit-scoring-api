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

    # Limites
    max_batch_size: int = 100
    max_request_timeout: int = 30  # secondes

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instance globale de configuration
settings = Settings()
