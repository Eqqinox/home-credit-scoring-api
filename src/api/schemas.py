"""
Schémas Pydantic pour la validation des données.

Définit les modèles de données pour les requêtes et réponses de l'API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Niveaux de risque possibles."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Decision(str, Enum):
    """Décisions possibles."""
    APPROVE = "approve"
    REFUSE = "refuse"


class ClientData(BaseModel):
    """
    Données d'un client pour la prédiction.

    Contient les 645 features nécessaires au modèle (sans TARGET).
    """

    # Features numériques principales (exemple - à compléter avec toutes les 645 features)
    SK_ID_CURR: Optional[int] = Field(None, description="ID du client")
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Revenu total du client")
    AMT_CREDIT: float = Field(..., gt=0, description="Montant du crédit")
    AMT_ANNUITY: Optional[float] = Field(None, ge=0, description="Annuité du prêt")
    AMT_GOODS_PRICE: Optional[float] = Field(None, ge=0, description="Prix des biens")

    # Features catégorielles
    NAME_CONTRACT_TYPE: str = Field(..., description="Type de contrat")
    CODE_GENDER: str = Field(..., description="Genre du client")
    FLAG_OWN_CAR: str = Field(..., description="Possède une voiture")
    FLAG_OWN_REALTY: str = Field(..., description="Possède un bien immobilier")

    # ... (Ajouter toutes les 645 features ici)
    # Pour l'instant, on accepte des features additionnelles via extra

    class Config:
        extra = "allow"  # Permet des champs supplémentaires
        json_schema_extra = {
            "example": {
                "SK_ID_CURR": 100001,
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y"
            }
        }

    @validator("AMT_INCOME_TOTAL")
    def validate_income(cls, v):
        """Valide que le revenu est dans une plage raisonnable."""
        if v > 1e9:  # 1 milliard
            raise ValueError("Le revenu semble anormalement élevé")
        return v

    @validator("AMT_CREDIT")
    def validate_credit(cls, v):
        """Valide que le montant du crédit est raisonnable."""
        if v > 1e9:
            raise ValueError("Le montant du crédit semble anormalement élevé")
        return v


class PredictionResponse(BaseModel):
    """Réponse de prédiction pour un client."""

    client_id: Optional[int] = Field(None, description="ID du client")
    probability_default: float = Field(..., ge=0.0, le=1.0, description="Probabilité de défaut")
    prediction: int = Field(..., ge=0, le=1, description="Prédiction binaire (0=pas de défaut, 1=défaut)")
    decision: Decision = Field(..., description="Décision finale (approve/refuse)")
    risk_level: RiskLevel = Field(..., description="Niveau de risque")
    threshold_used: float = Field(..., description="Seuil de décision utilisé")
    model_version: str = Field(..., description="Version du modèle")
    timestamp: datetime = Field(..., description="Horodatage de la prédiction")

    class Config:
        json_schema_extra = {
            "example": {
                "client_id": 100001,
                "probability_default": 0.23,
                "prediction": 0,
                "decision": "approve",
                "risk_level": "low",
                "threshold_used": 0.4955,
                "model_version": "1.0.0",
                "timestamp": "2025-12-01T10:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Requête pour des prédictions en batch."""

    clients: List[ClientData] = Field(..., max_length=100, description="Liste de clients (max 100)")

    @validator("clients")
    def validate_batch_size(cls, v):
        """Valide la taille du batch."""
        if len(v) == 0:
            raise ValueError("La liste de clients ne peut pas être vide")
        if len(v) > 100:
            raise ValueError("Maximum 100 clients par batch")
        return v


class BatchPredictionResponse(BaseModel):
    """Réponse pour des prédictions en batch."""

    predictions: List[PredictionResponse] = Field(..., description="Liste des prédictions")
    total_clients: int = Field(..., description="Nombre total de clients traités")
    timestamp: datetime = Field(..., description="Horodatage du batch")


class HealthResponse(BaseModel):
    """Réponse du health check."""

    status: str = Field(..., description="Statut de l'API")
    model_loaded: bool = Field(..., description="Le modèle est-il chargé")
    model_version: str = Field(..., description="Version du modèle")
    timestamp: datetime = Field(..., description="Horodatage")


class ModelInfoResponse(BaseModel):
    """Informations sur le modèle."""

    model_type: str = Field(..., description="Type de modèle")
    model_version: str = Field(..., description="Version du modèle")
    n_features: int = Field(..., description="Nombre de features")
    threshold: float = Field(..., description="Seuil de décision optimal")
    metrics: Dict[str, Any] = Field(..., description="Métriques du modèle")
    training_date: Optional[str] = Field(None, description="Date d'entraînement")


class ErrorResponse(BaseModel):
    """Réponse en cas d'erreur."""

    error: str = Field(..., description="Message d'erreur")
    detail: Optional[str] = Field(None, description="Détails de l'erreur")
    timestamp: datetime = Field(..., description="Horodatage")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Validation error",
                "detail": "Missing required feature: EXT_SOURCE_1",
                "timestamp": "2025-12-01T10:30:00"
            }
        }
