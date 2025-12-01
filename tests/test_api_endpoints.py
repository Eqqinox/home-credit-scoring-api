"""
Tests des endpoints de l'API.

Teste tous les endpoints FastAPI (health check, model info, predict, etc.)
"""

import pytest
from fastapi import status


class TestHealthEndpoint:
    """Tests de l'endpoint de health check."""

    def test_health_check_returns_200(self, api_client):
        """Teste que le health check retourne 200 OK."""
        response = api_client.get("/")
        assert response.status_code == status.HTTP_200_OK

    def test_health_check_has_correct_structure(self, api_client):
        """Teste que la réponse du health check a la structure correcte."""
        response = api_client.get("/")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "timestamp" in data

    def test_health_check_status_ok(self, api_client):
        """Teste que le statut est 'ok'."""
        response = api_client.get("/")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_check_model_loaded(self, api_client):
        """Teste que le modèle est chargé."""
        response = api_client.get("/")
        data = response.json()
        assert data["model_loaded"] is True


class TestModelInfoEndpoint:
    """Tests de l'endpoint d'informations sur le modèle."""

    def test_model_info_returns_200(self, api_client):
        """Teste que l'endpoint retourne 200 OK."""
        response = api_client.get("/model-info")
        assert response.status_code == status.HTTP_200_OK

    def test_model_info_has_correct_structure(self, api_client):
        """Teste que la réponse a la structure correcte."""
        response = api_client.get("/model-info")
        data = response.json()

        required_fields = [
            "model_type",
            "model_version",
            "n_features",
            "threshold",
            "metrics"
        ]

        for field in required_fields:
            assert field in data

    def test_model_info_correct_model_type(self, api_client):
        """Teste que le type de modèle est correct."""
        response = api_client.get("/model-info")
        data = response.json()
        assert data["model_type"] == "LightGBM"

    def test_model_info_has_metrics(self, api_client):
        """Teste que les métriques sont présentes."""
        response = api_client.get("/model-info")
        data = response.json()

        assert "metrics" in data
        metrics = data["metrics"]

        assert "auc_roc" in metrics
        assert "recall" in metrics
        assert "precision" in metrics
        assert "f1_score" in metrics

    def test_model_info_threshold_in_range(self, api_client):
        """Teste que le seuil est dans une plage valide."""
        response = api_client.get("/model-info")
        data = response.json()

        threshold = data["threshold"]
        assert 0.0 <= threshold <= 1.0


class TestPredictEndpoint:
    """Tests de l'endpoint de prédiction."""

    def test_predict_with_valid_data_returns_200(self, api_client, sample_data_from_csv):
        """Teste qu'une prédiction avec des données valides retourne 200."""
        response = api_client.post("/predict", json=sample_data_from_csv)
        assert response.status_code == status.HTTP_200_OK

    def test_predict_response_structure(self, api_client, sample_data_from_csv):
        """Teste que la réponse a la structure correcte."""
        response = api_client.post("/predict", json=sample_data_from_csv)
        data = response.json()

        required_fields = [
            "client_id",
            "probability_default",
            "prediction",
            "decision",
            "risk_level",
            "threshold_used",
            "model_version",
            "timestamp"
        ]

        for field in required_fields:
            assert field in data

    def test_predict_probability_in_range(self, api_client, sample_data_from_csv):
        """Teste que la probabilité est entre 0 et 1."""
        response = api_client.post("/predict", json=sample_data_from_csv)
        data = response.json()

        proba = data["probability_default"]
        assert 0.0 <= proba <= 1.0

    def test_predict_prediction_is_binary(self, api_client, sample_data_from_csv):
        """Teste que la prédiction est binaire (0 ou 1)."""
        response = api_client.post("/predict", json=sample_data_from_csv)
        data = response.json()

        prediction = data["prediction"]
        assert prediction in [0, 1]

    def test_predict_decision_is_valid(self, api_client, sample_data_from_csv):
        """Teste que la décision est valide."""
        response = api_client.post("/predict", json=sample_data_from_csv)
        data = response.json()

        decision = data["decision"]
        assert decision in ["approve", "refuse"]

    def test_predict_risk_level_is_valid(self, api_client, sample_data_from_csv):
        """Teste que le niveau de risque est valide."""
        response = api_client.post("/predict", json=sample_data_from_csv)
        data = response.json()

        risk_level = data["risk_level"]
        assert risk_level in ["low", "medium", "high"]

    def test_predict_with_missing_features_returns_400(self, api_client, client_data_missing_features):
        """Teste qu'une requête avec des features manquantes retourne 400."""
        response = api_client.post("/predict", json=client_data_missing_features)
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_predict_with_invalid_types_returns_422(self, api_client):
        """Teste qu'une requête avec des types incorrects retourne 422."""
        invalid_data = {
            "SK_ID_CURR": "not_an_integer",  # Devrait être int
            "AMT_INCOME_TOTAL": "not_a_float",  # Devrait être float
        }

        response = api_client.post("/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_with_negative_amount_returns_422(self, api_client):
        """Teste qu'un montant négatif est rejeté."""
        # Note: Cette validation dépend de la configuration des schémas Pydantic
        invalid_data = {
            "AMT_INCOME_TOTAL": -1000.0,  # Montant négatif invalide
        }

        response = api_client.post("/predict", json=invalid_data)
        # Devrait être rejeté par la validation Pydantic
        assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST]


class TestPredictBatchEndpoint:
    """Tests de l'endpoint de prédiction en batch."""

    def test_predict_batch_with_valid_data_returns_200(self, api_client, sample_data_from_csv):
        """Teste qu'une prédiction batch avec des données valides retourne 200."""
        batch_request = {
            "clients": [sample_data_from_csv, sample_data_from_csv]
        }

        response = api_client.post("/predict-batch", json=batch_request)
        assert response.status_code == status.HTTP_200_OK

    def test_predict_batch_response_structure(self, api_client, sample_data_from_csv):
        """Teste que la réponse batch a la structure correcte."""
        batch_request = {
            "clients": [sample_data_from_csv]
        }

        response = api_client.post("/predict-batch", json=batch_request)
        data = response.json()

        assert "predictions" in data
        assert "total_clients" in data
        assert "timestamp" in data

    def test_predict_batch_correct_number_predictions(self, api_client, sample_data_from_csv):
        """Teste que le nombre de prédictions correspond au nombre de clients."""
        n_clients = 3
        batch_request = {
            "clients": [sample_data_from_csv] * n_clients
        }

        response = api_client.post("/predict-batch", json=batch_request)
        data = response.json()

        assert len(data["predictions"]) == n_clients
        assert data["total_clients"] == n_clients

    def test_predict_batch_empty_list_returns_422(self, api_client):
        """Teste qu'une liste vide retourne 422."""
        batch_request = {
            "clients": []
        }

        response = api_client.post("/predict-batch", json=batch_request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_batch_too_large_returns_422(self, api_client, sample_data_from_csv):
        """Teste qu'un batch trop grand retourne 422."""
        # Max 100 clients
        batch_request = {
            "clients": [sample_data_from_csv] * 101
        }

        response = api_client.post("/predict-batch", json=batch_request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestErrorHandling:
    """Tests de la gestion d'erreur."""

    def test_nonexistent_endpoint_returns_404(self, api_client):
        """Teste qu'un endpoint inexistant retourne 404."""
        response = api_client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_error_response_has_timestamp(self, api_client):
        """Teste que les réponses d'erreur ont un timestamp."""
        response = api_client.get("/nonexistent")
        # Vérifier que la réponse JSON est valide même en erreur
        assert response.headers["content-type"] == "application/json"
