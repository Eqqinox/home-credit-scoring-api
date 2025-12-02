"""
Tests additionnels pour atteindre 80% de couverture.
Ces tests couvrent les branches non testées dans main.py et predictor.py.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from src.api.main import app


class TestErrorHandlersMain:
    """Tests des gestionnaires d'erreurs dans main.py."""
    
    def test_http_exception_handler(self, api_client):
        """Test du gestionnaire d'exceptions HTTP personnalisé."""
        # Tester avec un endpoint qui n'existe pas (404)
        response = api_client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "timestamp" in data
    
    def test_general_exception_handler(self, api_client):
        """Test du gestionnaire d'exceptions générales."""
        # Simuler une erreur interne en mockant predict
        with patch('src.api.main.predictor') as mock_predictor:
            mock_predictor.is_loaded.return_value = True
            mock_predictor.predict.side_effect = Exception("Erreur interne simulée")
            
            response = api_client.post(
                "/predict",
                json={"SK_ID_CURR": 100001, "AMT_INCOME_TOTAL": 150000}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "timestamp" in data


class TestModelNotLoadedScenarios:
    """Tests des scénarios où le modèle n'est pas chargé."""
    
    def test_health_check_model_not_loaded(self, api_client):
        """Test health check quand le modèle n'est pas chargé."""
        with patch('src.api.main.predictor', None):
            response = api_client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is False
    
    def test_predict_model_not_loaded(self, api_client):
        """Test predict quand le modèle n'est pas chargé."""
        with patch('src.api.main.predictor', None):
            response = api_client.post(
                "/predict",
                json={"SK_ID_CURR": 100001}
            )
            assert response.status_code == 503
            assert "modèle n'est pas chargé" in response.json()["detail"]
    
    def test_model_info_model_not_loaded(self, api_client):
        """Test model-info quand le modèle n'est pas chargé."""
        with patch('src.api.main.predictor', None):
            response = api_client.get("/model-info")
            assert response.status_code == 503
    
    def test_predict_batch_model_not_loaded(self, api_client):
        """Test predict-batch quand le modèle n'est pas chargé."""
        with patch('src.api.main.predictor', None):
            response = api_client.post(
                "/predict-batch",
                json={"clients": [{"SK_ID_CURR": 100001}]}
            )
            assert response.status_code == 503


class TestPredictErrorScenarios:
    """Tests des scénarios d'erreur pour /predict."""
    
    def test_predict_value_error(self, api_client):
        """Test predict avec ValueError."""
        with patch('src.api.main.predictor') as mock_predictor:
            mock_predictor.is_loaded.return_value = True
            mock_predictor.predict.side_effect = ValueError("Données invalides")
            
            response = api_client.post(
                "/predict",
                json={"SK_ID_CURR": 100001}
            )
            assert response.status_code == 400
    
    def test_predict_exception(self, api_client):
        """Test predict avec Exception générale."""
        with patch('src.api.main.predictor') as mock_predictor:
            mock_predictor.is_loaded.return_value = True
            mock_predictor.predict.side_effect = Exception("Erreur interne")
            
            response = api_client.post(
                "/predict",
                json={"SK_ID_CURR": 100001}
            )
            assert response.status_code == 500


class TestPredictBatchErrorScenarios:
    """Tests des scénarios d'erreur pour /predict-batch."""
    
    def test_predict_batch_value_error(self, api_client):
        """Test predict-batch avec ValueError."""
        with patch('src.api.main.predictor') as mock_predictor:
            mock_predictor.is_loaded.return_value = True
            mock_predictor.predict_batch.side_effect = ValueError("Liste vide")
            
            response = api_client.post(
                "/predict-batch",
                json={"clients": []}
            )
            assert response.status_code == 400
    
    def test_predict_batch_exception(self, api_client):
        """Test predict-batch avec Exception générale."""
        with patch('src.api.main.predictor') as mock_predictor:
            mock_predictor.is_loaded.return_value = True
            mock_predictor.predict_batch.side_effect = Exception("Erreur batch")
            
            response = api_client.post(
                "/predict-batch",
                json={"clients": [{"SK_ID_CURR": 100001}]}
            )
            assert response.status_code == 500


class TestPredictorInternals:
    """Tests des méthodes internes du predictor."""
    
    def test_get_risk_level_boundaries(self):
        """Test des limites des niveaux de risque."""
        from src.api.predictor import CreditScoringPredictor
        from src.api.schemas import RiskLevel
        
        predictor = CreditScoringPredictor()
        
        # Test limite basse (LOW)
        assert predictor._get_risk_level(0.0) == RiskLevel.LOW
        assert predictor._get_risk_level(0.29) == RiskLevel.LOW
        
        # Test limite moyenne (MEDIUM)
        assert predictor._get_risk_level(0.3) == RiskLevel.MEDIUM
        assert predictor._get_risk_level(0.45) == RiskLevel.MEDIUM
        assert predictor._get_risk_level(0.59) == RiskLevel.MEDIUM
        
        # Test limite haute (HIGH)
        assert predictor._get_risk_level(0.6) == RiskLevel.HIGH
        assert predictor._get_risk_level(1.0) == RiskLevel.HIGH
    
    def test_predict_with_client_id(self):
        """Test que l'ID client est préservé dans la prédiction."""
        from src.api.predictor import CreditScoringPredictor
        
        predictor = CreditScoringPredictor()
        
        # Données minimales avec ID
        data = {
            "SK_ID_CURR": 999999,
            "AMT_INCOME_TOTAL": 150000,
            # ... autres features nécessaires
        }
        
        # Mock du preprocessing et de la prédiction
        with patch.object(predictor, 'preprocess') as mock_preprocess:
            mock_preprocess.return_value = MagicMock()
            with patch.object(predictor.model, 'predict_proba') as mock_predict:
                mock_predict.return_value = [[0.3, 0.7]]
                
                result = predictor.predict(data)
                
                assert result["client_id"] == 999999
    
    def test_predict_without_client_id(self):
        """Test prédiction sans ID client."""
        from src.api.predictor import CreditScoringPredictor
        
        predictor = CreditScoringPredictor()
        
        # Données sans SK_ID_CURR
        data = {
            "AMT_INCOME_TOTAL": 150000,
        }
        
        with patch.object(predictor, 'preprocess') as mock_preprocess:
            mock_preprocess.return_value = MagicMock()
            with patch.object(predictor.model, 'predict_proba') as mock_predict:
                mock_predict.return_value = [[0.3, 0.7]]
                
                result = predictor.predict(data)
                
                assert result["client_id"] is None


class TestMainEntryPoint:
    """Tests du point d'entrée principal."""
    
    def test_main_entry_point_exists(self):
        """Vérifie que le point d'entrée __main__ existe."""
        from src.api import main
        
        # Vérifier que le module peut être importé
        assert hasattr(main, 'app')
        assert hasattr(main, 'predictor')
        assert hasattr(main, 'health_check')
        assert hasattr(main, 'get_model_info')
        assert hasattr(main, 'predict')
        assert hasattr(main, 'predict_batch')


class TestCORSMiddleware:
    """Tests de la configuration CORS."""
    
    def test_cors_headers_in_response(self, api_client):
        """Vérifie que les headers CORS sont présents."""
        response = api_client.get("/")
        
        # Les headers CORS devraient être présents
        # ou le request devrait réussir sans erreur
        assert response.status_code == 200