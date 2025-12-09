"""
Tests pour le stockage PostgreSQL des prédictions.
"""
import pytest
from src.monitoring.storage import PredictionStorage
import uuid


@pytest.fixture(scope="module")
def storage():
    """Fixture pour initialiser PredictionStorage avec DB de test."""
    test_db_url = "postgresql://moon:moon@localhost:5432/credit_scoring_test"
    storage = PredictionStorage(database_url=test_db_url)
    yield storage
    storage.close()


def test_save_prediction_success(storage):
    """Test l'enregistrement d'une prédiction valide."""
    request_id = str(uuid.uuid4())
    prediction_data = {
        'client_id': 100001,
        'probability_default': 0.35,
        'prediction': 0,
        'decision': 'APPROVE',
        'risk_level': 'MEDIUM',
        'threshold_used': 0.5225,
        'model_version': '1.0.0'
    }
    timing_data = {'preprocessing_ms': 15.3, 'inference_ms': 8.7, 'total_ms': 24.0}
    input_features = {
        'AMT_CREDIT': 406597.5,
        'AMT_INCOME_TOTAL': 202500.0,
        'NAME_CONTRACT_TYPE': 'Cash loans',
        'FLAG_OWN_CAR': 'N'
    }

    success = storage.save_prediction(
        request_id=request_id,
        endpoint="/predict",
        prediction_data=prediction_data,
        timing_data=timing_data,
        input_features=input_features,
        api_version="1.0.0",
        http_status=200
    )

    assert success is True


def test_get_predictions(storage):
    """Test la récupération des prédictions."""
    predictions = storage.get_predictions(limit=10)
    assert isinstance(predictions, list)
    if len(predictions) > 0:
        pred = predictions[0]
        assert 'request_id' in pred
        assert 'timestamp' in pred
        assert 'decision' in pred


def test_get_stats(storage):
    """Test le calcul des statistiques."""
    stats = storage.get_stats()
    assert isinstance(stats, dict)
    assert 'total_predictions' in stats
    assert 'approval_rate' in stats
    assert 'avg_inference_time_ms' in stats


def test_confidence_level_calculation(storage):
    """Test le calcul du confidence_level."""
    # Probabilité proche du seuil (0.5225)
    assert storage._calculate_confidence_level(0.53) == 'LOW'
    # Probabilité moyenne distance
    assert storage._calculate_confidence_level(0.65) == 'MEDIUM'
    # Probabilité loin du seuil
    assert storage._calculate_confidence_level(0.85) == 'HIGH'


def test_data_quality_score_calculation(storage):
    """Test le calcul du data_quality_score."""
    features_complete = {
        'EXT_SOURCE_1': 0.5,
        'EXT_SOURCE_2': 0.6,
        'EXT_SOURCE_3': 0.7,
        'AMT_CREDIT': 100000,
        'AMT_INCOME_TOTAL': 50000
    }
    score = storage._calculate_data_quality_score(features_complete)
    assert 0 <= score <= 1
    assert score > 0.8  # Score élevé car features critiques présentes
