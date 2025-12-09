"""
Point d'entrée de l'API FastAPI.

Définit tous les endpoints de l'API de scoring crédit.
"""

import logging
import uuid
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from .config import settings
from .schemas import (
    ClientData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from .predictor import CreditScoringPredictor
from src.monitoring.logger import configure_structlog, ProductionLogger
from src.monitoring.storage import PredictionStorage

# Configuration du logging structuré
configure_structlog(
    log_level=settings.log_level,
    environment=settings.environment
)

# Configuration du logging standard (pour compatibilité)
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Instance globale du logger de production
production_logger = ProductionLogger()

# Instance globale du predictor (chargée au démarrage)
predictor: CreditScoringPredictor = None
storage: PredictionStorage = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application.

    Charge le modèle au démarrage et nettoie à l'arrêt.
    """
    # Démarrage
    global predictor, storage
    logger.info("Démarrage de l'application")

    # 1. Charger le modèle
    logger.info("Chargement du modèle et des encoders...")
    try:
        predictor = CreditScoringPredictor()
        logger.info("Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur fatale lors du chargement du modèle: {e}")
        raise

    # 2. Initialiser PredictionStorage
    if settings.db_store_predictions:
        logger.info("Initialisation de PredictionStorage...")
        try:
            storage = PredictionStorage(
                database_url=settings.database_url,
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow
            )
            logger.info("PredictionStorage initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du storage: {e}")
            logger.warning("L'API continuera sans stockage PostgreSQL")
            storage = None

    yield

    # Arrêt
    logger.info("Arrêt de l'application")
    if storage:
        try:
            storage.close()
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture du storage: {e}")


# Création de l'application FastAPI
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan
)

# Configuration CORS (pour permettre les requêtes cross-origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== ENDPOINTS =====

@app.get(
    "/",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Vérifie que l'API fonctionne et que le modèle est chargé"
)
async def health_check() -> HealthResponse:
    """
    Endpoint de health check.

    Returns:
        Statut de l'API et du modèle
    """
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded() if predictor else False,
        model_version=settings.api_version,
        timestamp=datetime.now()
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Informations sur le modèle",
    description="Retourne les informations et métriques du modèle"
)
async def get_model_info() -> ModelInfoResponse:
    """
    Retourne les informations sur le modèle.

    Returns:
        Informations complètes sur le modèle (type, version, métriques, etc.)

    Raises:
        HTTPException: Si le modèle n'est pas chargé
    """
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas chargé"
        )

    model_info = predictor.get_model_info()
    return ModelInfoResponse(**model_info)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Prédiction pour un client",
    description="Fait une prédiction de risque de défaut pour un client",
    responses={
        200: {"description": "Prédiction réussie"},
        400: {"description": "Données invalides", "model": ErrorResponse},
        422: {"description": "Validation échouée", "model": ErrorResponse},
        500: {"description": "Erreur serveur", "model": ErrorResponse}
    }
)
async def predict(client_data: ClientData) -> PredictionResponse:
    """
    Fait une prédiction pour un client.

    Args:
        client_data: Données du client (645 features)

    Returns:
        Prédiction avec probabilité, décision et niveau de risque

    Raises:
        HTTPException: Si la prédiction échoue
    """
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas chargé"
        )

    request_id = str(uuid.uuid4())

    try:
        # Convertir le modèle Pydantic en dictionnaire
        data_dict = client_data.model_dump()

        # Faire la prédiction
        result = predictor.predict(data_dict)

        # Extraire les timings (ajoutés par predictor)
        timing = result.pop('_timing', {})

        # Logger la prédiction avec structlog
        production_logger.log_prediction(
            client_id=result.get('client_id'),
            probability=result['probability_default'],
            prediction=result['prediction'],
            decision=result['decision'],
            risk_level=result['risk_level'],
            preprocessing_time_ms=timing.get('preprocessing_ms', 0),
            inference_time_ms=timing.get('inference_ms', 0),
            total_time_ms=timing.get('total_ms', 0),
            threshold=result['threshold_used'],
            model_version=result['model_version'],
            endpoint="/predict",
            http_status=200,
        )

        # Stocker en PostgreSQL
        if storage:
            try:
                storage.save_prediction(
                    request_id=request_id,
                    endpoint="/predict",
                    prediction_data=result,
                    timing_data=timing,
                    input_features=data_dict,
                    api_version=settings.api_version,
                    http_status=200
                )
            except Exception as db_error:
                logger.error(f"Erreur lors du stockage PostgreSQL: {db_error}")
                # L'API continue même si le stockage échoue

        return PredictionResponse(**result)

    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        production_logger.log_error(
            error_type="ValidationError",
            error_message=str(e),
            endpoint="/predict",
            client_id=client_data.SK_ID_CURR,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        production_logger.log_error(
            error_type="InternalError",
            error_message=str(e),
            endpoint="/predict",
            client_id=client_data.SK_ID_CURR,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Prédictions en batch",
    description=f"Fait des prédictions pour plusieurs clients (max {settings.max_batch_size})",
    responses={
        200: {"description": "Prédictions réussies"},
        400: {"description": "Données invalides", "model": ErrorResponse},
        422: {"description": "Validation échouée", "model": ErrorResponse},
        500: {"description": "Erreur serveur", "model": ErrorResponse}
    }
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Fait des prédictions pour plusieurs clients.

    Args:
        request: Liste de clients (max 100)

    Returns:
        Liste de prédictions

    Raises:
        HTTPException: Si les prédictions échouent
    """
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas chargé"
        )

    # Démarrer le chronomètre pour le batch
    import time
    start_batch_time = time.time()

    try:
        # Convertir les modèles Pydantic en dictionnaires
        clients_data = [client.model_dump() for client in request.clients]

        # Faire les prédictions
        predictions = predictor.predict_batch(clients_data)

        # Stocker chaque prédiction en PostgreSQL
        if storage:
            for i, pred in enumerate(predictions):
                try:
                    request_id = str(uuid.uuid4())
                    timing = pred.get('_timing', {})

                    storage.save_prediction(
                        request_id=request_id,
                        endpoint="/predict-batch",
                        prediction_data=pred,
                        timing_data=timing,
                        input_features=clients_data[i],
                        api_version=settings.api_version,
                        http_status=200
                    )
                except Exception as db_error:
                    logger.error(f"Erreur lors du stockage batch {i}: {db_error}")
                    # Continue même en cas d'erreur

        # Retirer les _timing de chaque prédiction
        for pred in predictions:
            pred.pop('_timing', None)

        # Convertir en PredictionResponse
        prediction_responses = [PredictionResponse(**pred) for pred in predictions]

        # Calculer le temps total
        total_time_ms = (time.time() - start_batch_time) * 1000

        # Logger le batch
        production_logger.log_batch(
            n_clients=len(request.clients),
            total_time_ms=total_time_ms,
            n_success=len(prediction_responses),
            n_errors=0,
        )

        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_clients=len(prediction_responses),
            timestamp=datetime.now()
        )

    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        production_logger.log_error(
            error_type="ValidationError",
            error_message=str(e),
            endpoint="/predict-batch",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erreur lors des prédictions batch: {e}")
        production_logger.log_error(
            error_type="InternalError",
            error_message=str(e),
            endpoint="/predict-batch",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors des prédictions: {str(e)}"
        )


# ===== GESTION DES ERREURS =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handler personnalisé pour les erreurs HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": None,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handler pour toutes les autres erreurs."""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Erreur interne du serveur",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ===== POINT D'ENTRÉE =====

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
