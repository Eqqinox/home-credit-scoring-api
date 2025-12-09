"""
Stockage PostgreSQL des prédictions.
"""
from sqlalchemy import Column, String, Integer, Float, TIMESTAMP, Boolean, TEXT, ARRAY, ForeignKey, CheckConstraint, Index, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import uuid

logger = logging.getLogger(__name__)

Base = declarative_base()

# ===== MODÈLES ORM =====

class Prediction(Base):
    """Modèle pour la table predictions."""
    __tablename__ = "predictions"

    request_id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)
    endpoint = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)
    api_version = Column(String(20), nullable=False)

    preprocessing_time_ms = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=False)
    total_response_time_ms = Column(Float, nullable=False)
    memory_usage_mb = Column(Float, nullable=True)

    prediction_proba = Column(Float, nullable=False)
    prediction_class = Column(Integer, nullable=False)
    decision = Column(String(20), nullable=False)
    confidence_level = Column(String(10), nullable=True)

    http_status_code = Column(Integer, nullable=False)
    error_message = Column(TEXT, nullable=True)
    data_quality_score = Column(Float, nullable=True)
    warning_flags = Column(ARRAY(TEXT), nullable=True)

    client_id = Column(String(50), nullable=True)
    loan_amount = Column(Float, nullable=True)
    decision_threshold = Column(Float, nullable=False, default=0.5225)

    feature_values = relationship("FeatureValue", back_populates="prediction", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint('prediction_proba BETWEEN 0 AND 1', name='chk_proba'),
        CheckConstraint('prediction_class IN (0, 1)', name='chk_class'),
        Index('idx_timestamp', 'timestamp'),
        Index('idx_decision', 'decision'),
        Index('idx_status_code', 'http_status_code'),
    )

class FeatureValue(Base):
    """Modèle pour la table feature_values."""
    __tablename__ = "feature_values"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(PGUUID(as_uuid=True), ForeignKey('predictions.request_id', ondelete='CASCADE'), nullable=False)
    feature_name = Column(String(100), nullable=False)
    feature_value = Column(Float, nullable=True)
    is_null = Column(Boolean, default=False)

    prediction = relationship("Prediction", back_populates="feature_values")

    __table_args__ = (Index('idx_feature_name_time', 'feature_name', 'request_id'),)

class Anomaly(Base):
    """Modèle pour la table anomalies."""
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(PGUUID(as_uuid=True), ForeignKey('predictions.request_id'), nullable=True)
    detected_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)
    anomaly_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    description = Column(TEXT, nullable=False)
    affected_features = Column(ARRAY(TEXT), nullable=True)
    metric_value = Column(Float, nullable=True)
    threshold_value = Column(Float, nullable=True)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(TIMESTAMP, nullable=True)
    resolution_notes = Column(TEXT, nullable=True)

    __table_args__ = (
        CheckConstraint("severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')", name='chk_severity'),
        Index('idx_anomaly_type', 'anomaly_type'),
        Index('idx_severity', 'severity'),
        Index('idx_detected_at', 'detected_at'),
    )

class DriftReport(Base):
    """Modèle pour la table drift_reports."""
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generated_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)
    report_period_start = Column(TIMESTAMP, nullable=False)
    report_period_end = Column(TIMESTAMP, nullable=False)
    reference_dataset = Column(String(100), nullable=True)
    current_dataset_size = Column(Integer, nullable=False)
    reference_dataset_size = Column(Integer, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    drift_score = Column(Float, nullable=True)
    n_features_drifted = Column(Integer, nullable=True)
    drifted_features = Column(ARRAY(TEXT), nullable=True)
    report_html_path = Column(TEXT, nullable=False)
    report_json_path = Column(TEXT, nullable=True)
    metadata_json = Column('metadata', JSONB, nullable=True)

    __table_args__ = (Index('idx_report_generated_at', 'generated_at'),)

# ===== CLASSE PRINCIPALE =====

class PredictionStorage:
    """
    Classe de gestion du stockage PostgreSQL des prédictions.

    Méthodes principales :
    - save_prediction() : Enregistre une prédiction avec ses features
    - get_predictions() : Récupère les prédictions avec filtres
    - get_stats() : Calcule des statistiques sur les prédictions
    """

    # Top 20 features à stocker
    TOP_20_FEATURES = [
        "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "FLAG_MOBIL",
        "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"
    ]

    def __init__(self, database_url: str, pool_size: int = 5, max_overflow: int = 10):
        """
        Initialise la connexion PostgreSQL.

        Args:
            database_url: URL de connexion PostgreSQL
            pool_size: Taille du pool de connexions
            max_overflow: Connexions supplémentaires autorisées
        """
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._initialize_engine()

    def _initialize_engine(self):
        """Crée le moteur SQLAlchemy avec pool de connexions."""
        try:
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=30,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False
            )

            # Test connexion
            with self.engine.connect() as conn:
                logger.info("Connexion PostgreSQL établie avec succès")

            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("Pool de connexions PostgreSQL initialisé")

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
            raise

    def close(self):
        """Ferme le pool de connexions."""
        if self.engine:
            logger.info("Fermeture du pool de connexions PostgreSQL")
            self.engine.dispose()

    @contextmanager
    def get_session(self) -> Session:
        """Context manager pour obtenir une session DB."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur lors de la transaction DB: {e}")
            raise
        finally:
            session.close()

    def save_prediction(
        self,
        request_id: str,
        endpoint: str,
        prediction_data: Dict[str, Any],
        timing_data: Dict[str, float],
        input_features: Dict[str, Any],
        api_version: str,
        http_status: int = 200,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Enregistre une prédiction en base de données.

        Args:
            request_id: UUID de la requête
            endpoint: Endpoint API appelé
            prediction_data: Résultat de la prédiction
            timing_data: Temps de traitement
            input_features: Features d'entrée (top 20 seront extraites)
            api_version: Version de l'API
            http_status: Code HTTP de la réponse
            error_message: Message d'erreur si échec

        Returns:
            True si succès, False si échec
        """
        try:
            with self.get_session() as session:
                # Calculer confidence_level et data_quality_score
                confidence_level = self._calculate_confidence_level(prediction_data['probability_default'])
                data_quality_score = self._calculate_data_quality_score(input_features)

                client_id = str(prediction_data.get('client_id')) if prediction_data.get('client_id') else None
                loan_amount = input_features.get('AMT_CREDIT')

                # Créer l'objet Prediction
                prediction = Prediction(
                    request_id=request_id,
                    timestamp=datetime.utcnow(),
                    endpoint=endpoint,
                    model_version=prediction_data['model_version'],
                    api_version=api_version,
                    preprocessing_time_ms=timing_data.get('preprocessing_ms'),
                    inference_time_ms=timing_data['inference_ms'],
                    total_response_time_ms=timing_data['total_ms'],
                    memory_usage_mb=None,
                    prediction_proba=prediction_data['probability_default'],
                    prediction_class=prediction_data['prediction'],
                    decision=prediction_data['decision'],
                    confidence_level=confidence_level,
                    http_status_code=http_status,
                    error_message=error_message,
                    data_quality_score=data_quality_score,
                    warning_flags=None,
                    client_id=client_id,
                    loan_amount=loan_amount,
                    decision_threshold=prediction_data['threshold_used'],
                )

                session.add(prediction)
                session.flush()

                # Ajouter les feature_values (top 20)
                feature_values = self._extract_top_features(input_features, request_id)
                for fv in feature_values:
                    session.add(fv)

                logger.info(f"Prédiction {request_id} enregistrée en DB")
                return True

        except SQLAlchemyError as e:
            logger.error(f"Erreur SQL lors de l'insertion de la prédiction {request_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion de la prédiction {request_id}: {e}")
            return False

    def get_predictions(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        decision: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère les prédictions avec filtres optionnels.

        Args:
            limit: Nombre max de résultats
            offset: Décalage pour pagination
            start_date: Date de début (filtre)
            end_date: Date de fin (filtre)
            decision: Filtre par décision (APPROVE/REFUSE)

        Returns:
            Liste de dictionnaires contenant les prédictions
        """
        try:
            with self.get_session() as session:
                query = session.query(Prediction)

                if start_date:
                    query = query.filter(Prediction.timestamp >= start_date)
                if end_date:
                    query = query.filter(Prediction.timestamp <= end_date)
                if decision:
                    query = query.filter(Prediction.decision == decision)

                predictions = query.order_by(Prediction.timestamp.desc()).limit(limit).offset(offset).all()

                return [
                    {
                        'request_id': str(pred.request_id),
                        'timestamp': pred.timestamp.isoformat(),
                        'client_id': pred.client_id,
                        'prediction_proba': pred.prediction_proba,
                        'prediction_class': pred.prediction_class,
                        'decision': pred.decision,
                        'confidence_level': pred.confidence_level,
                        'http_status_code': pred.http_status_code,
                        'total_response_time_ms': pred.total_response_time_ms
                    }
                    for pred in predictions
                ]

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des prédictions: {e}")
            return []

    def get_stats(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les prédictions.

        Args:
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)

        Returns:
            Dictionnaire de statistiques
        """
        try:
            with self.get_session() as session:
                query = session.query(Prediction)

                if start_date:
                    query = query.filter(Prediction.timestamp >= start_date)
                if end_date:
                    query = query.filter(Prediction.timestamp <= end_date)

                total_predictions = query.count()
                approve_count = query.filter(Prediction.decision == 'approve').count()
                refuse_count = query.filter(Prediction.decision == 'refuse').count()
                error_count = query.filter(Prediction.http_status_code != 200).count()

                avg_inference_time = session.query(Prediction.inference_time_ms).filter(
                    Prediction.http_status_code == 200
                ).all()
                avg_inference_time_ms = sum([t[0] for t in avg_inference_time]) / len(avg_inference_time) if avg_inference_time else 0

                return {
                    'total_predictions': total_predictions,
                    'approve_count': approve_count,
                    'refuse_count': refuse_count,
                    'error_count': error_count,
                    'approval_rate': approve_count / total_predictions if total_predictions > 0 else 0,
                    'avg_inference_time_ms': round(avg_inference_time_ms, 2)
                }

        except Exception as e:
            logger.error(f"Erreur lors du calcul des statistiques: {e}")
            return {}

    # ===== MÉTHODES PRIVÉES =====

    def _calculate_confidence_level(self, probability: float) -> str:
        """Calcule le niveau de confiance basé sur la distance au seuil."""
        threshold = 0.5225
        distance = abs(probability - threshold)

        if distance < 0.05:
            return 'LOW'
        elif distance < 0.15:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _calculate_data_quality_score(self, features: Dict[str, Any]) -> float:
        """Calcule un score de qualité des données (0-1)."""
        total = len(features)
        non_null = sum(1 for v in features.values() if v is not None and v != '')
        basic_score = non_null / total if total > 0 else 0

        critical_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']
        critical_present = sum(1 for f in critical_features if features.get(f) is not None)
        critical_bonus = critical_present / len(critical_features) * 0.2

        return round(min(basic_score + critical_bonus, 1.0), 3)

    def _extract_top_features(self, features: Dict[str, Any], request_id: str) -> List[FeatureValue]:
        """Extrait les top 20 features pour le stockage."""
        feature_values = []

        for feature_name in self.TOP_20_FEATURES:
            value = features.get(feature_name)

            if value is not None:
                try:
                    value = float(value)
                    is_null = False
                except (ValueError, TypeError):
                    value = None
                    is_null = True
            else:
                is_null = True

            feature_values.append(
                FeatureValue(
                    request_id=request_id,
                    feature_name=feature_name,
                    feature_value=value,
                    is_null=is_null
                )
            )

        return feature_values
