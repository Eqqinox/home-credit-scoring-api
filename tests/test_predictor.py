"""
Tests de la classe CreditScoringPredictor.

Teste le chargement du modèle, le preprocessing, les prédictions
et la gestion des erreurs.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestPredictorLoading:
    """Tests du chargement des artefacts."""

    def test_predictor_initialization(self, predictor):
        """Teste que le predictor s'initialise correctement."""
        assert predictor is not None
        assert predictor.is_loaded()

    def test_model_loaded(self, predictor):
        """Teste que le modèle est chargé."""
        assert predictor.model is not None
        assert hasattr(predictor.model, 'predict_proba')

    def test_label_encoders_loaded(self, predictor):
        """Teste que les label encoders sont chargés."""
        assert predictor.label_encoders is not None
        assert isinstance(predictor.label_encoders, dict)
        assert len(predictor.label_encoders) > 0

    def test_onehot_encoders_loaded(self, predictor):
        """Teste que les one-hot encoders sont chargés."""
        assert predictor.onehot_encoders is not None
        assert isinstance(predictor.onehot_encoders, dict)
        assert len(predictor.onehot_encoders) > 0

    def test_feature_names_loaded(self, predictor):
        """Teste que les noms de features sont chargés."""
        assert predictor.feature_names is not None
        assert isinstance(predictor.feature_names, list)
        assert len(predictor.feature_names) > 0

    def test_threshold_loaded(self, predictor):
        """Teste que le seuil optimal est chargé."""
        assert predictor.threshold is not None
        assert isinstance(predictor.threshold, float)
        assert 0.0 < predictor.threshold < 1.0

    def test_threshold_value_correct(self, predictor):
        """Teste que le seuil optimal est dans une plage raisonnable."""
        # Le seuil optimal varie légèrement selon l'entraînement
        # Actuellement : 0.5225 (modèle sans SK_ID_CURR)
        # Anciennement : 0.4955 (modèle avec SK_ID_CURR)
        assert 0.49 <= predictor.threshold <= 0.53

    def test_metrics_loaded(self, predictor):
        """Teste que les métriques sont chargées."""
        assert predictor.metrics is not None
        assert isinstance(predictor.metrics, dict)

        required_metrics = ['auc_roc', 'recall', 'precision', 'f1_score']
        for metric in required_metrics:
            assert metric in predictor.metrics


class TestPreprocessing:
    """Tests du preprocessing des données."""

    def test_preprocessing_returns_dataframe(self, predictor, sample_data_from_csv):
        """Teste que le preprocessing retourne un DataFrame."""
        result = predictor.preprocess(sample_data_from_csv)
        assert isinstance(result, pd.DataFrame)

    def test_preprocessing_correct_shape(self, predictor, sample_data_from_csv):
        """Teste que le DataFrame a la bonne forme."""
        result = predictor.preprocess(sample_data_from_csv)

        # Une seule observation
        assert result.shape[0] == 1

        # Nombre de features attendu
        assert result.shape[1] == len(predictor.feature_names)

    def test_preprocessing_feature_order(self, predictor, sample_data_from_csv):
        """Teste que les features sont dans le bon ordre."""
        result = predictor.preprocess(sample_data_from_csv)
        assert list(result.columns) == predictor.feature_names

    def test_preprocessing_removes_sk_id_curr(self, predictor, sample_data_from_csv):
        """Teste que SK_ID_CURR est retiré."""
        result = predictor.preprocess(sample_data_from_csv)

        # SK_ID_CURR ne doit pas être dans les colonnes
        assert 'SK_ID_CURR' not in result.columns

    def test_preprocessing_handles_unknown_category(self, predictor, sample_data_from_csv):
        """Teste la gestion des catégories inconnues."""
        # Modifier une catégorie pour la rendre inconnue
        data = sample_data_from_csv.copy()
        data['NAME_CONTRACT_TYPE'] = 'Unknown_Category_XXX'

        # Le preprocessing doit fonctionner sans erreur
        result = predictor.preprocess(data)
        assert result is not None
        assert result.shape[0] == 1

    def test_preprocessing_no_missing_values(self, predictor, sample_data_from_csv):
        """Teste qu'il n'y a pas de valeurs manquantes après preprocessing."""
        result = predictor.preprocess(sample_data_from_csv)
        assert result.isna().sum().sum() == 0

    def test_preprocessing_numeric_types(self, predictor, sample_data_from_csv):
        """Teste que toutes les valeurs sont numériques."""
        result = predictor.preprocess(sample_data_from_csv)

        # Toutes les colonnes doivent être numériques
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])


class TestPredictions:
    """Tests des prédictions."""

    def test_predict_returns_dict(self, predictor, sample_data_from_csv):
        """Teste que predict retourne un dictionnaire."""
        result = predictor.predict(sample_data_from_csv)
        assert isinstance(result, dict)

    def test_predict_has_required_fields(self, predictor, sample_data_from_csv):
        """Teste que la prédiction contient tous les champs requis."""
        result = predictor.predict(sample_data_from_csv)

        required_fields = [
            'client_id',
            'probability_default',
            'prediction',
            'decision',
            'risk_level',
            'threshold_used',
            'model_version',
            'timestamp'
        ]

        for field in required_fields:
            assert field in result

    def test_predict_probability_in_range(self, predictor, sample_data_from_csv):
        """Teste que la probabilité est entre 0 et 1."""
        result = predictor.predict(sample_data_from_csv)
        proba = result['probability_default']

        assert isinstance(proba, float)
        assert 0.0 <= proba <= 1.0

    def test_predict_prediction_is_binary(self, predictor, sample_data_from_csv):
        """Teste que la prédiction est binaire."""
        result = predictor.predict(sample_data_from_csv)
        prediction = result['prediction']

        assert prediction in [0, 1]

    def test_predict_decision_is_valid(self, predictor, sample_data_from_csv):
        """Teste que la décision est valide."""
        result = predictor.predict(sample_data_from_csv)
        decision = result['decision']

        assert decision in ['approve', 'refuse']

    def test_predict_risk_level_is_valid(self, predictor, sample_data_from_csv):
        """Teste que le niveau de risque est valide."""
        result = predictor.predict(sample_data_from_csv)
        risk_level = result['risk_level']

        assert risk_level in ['low', 'medium', 'high']

    def test_predict_threshold_applied_correctly(self, predictor, sample_data_from_csv):
        """Teste que le seuil optimal est appliqué correctement."""
        result = predictor.predict(sample_data_from_csv)

        proba = result['probability_default']
        prediction = result['prediction']
        threshold = result['threshold_used']

        # Vérifier que le seuil utilisé est le bon
        assert threshold == predictor.threshold

        # Vérifier que la prédiction correspond au seuil
        if proba >= threshold:
            assert prediction == 1
        else:
            assert prediction == 0

    def test_predict_decision_matches_prediction(self, predictor, sample_data_from_csv):
        """Teste que la décision correspond à la prédiction."""
        result = predictor.predict(sample_data_from_csv)

        prediction = result['prediction']
        decision = result['decision']

        if prediction == 1:
            assert decision == 'refuse'
        else:
            assert decision == 'approve'

    def test_predict_risk_level_low(self, predictor, sample_data_from_csv):
        """Teste la classification en risque faible."""
        # On ne peut pas forcer la probabilité, mais on peut vérifier la logique
        result = predictor.predict(sample_data_from_csv)

        proba = result['probability_default']
        risk_level = result['risk_level']

        if proba < 0.3:
            assert risk_level == 'low'

    def test_predict_risk_level_medium(self, predictor, sample_data_from_csv):
        """Teste la classification en risque moyen."""
        result = predictor.predict(sample_data_from_csv)

        proba = result['probability_default']
        risk_level = result['risk_level']

        if 0.3 <= proba < 0.6:
            assert risk_level == 'medium'

    def test_predict_risk_level_high(self, predictor, sample_data_from_csv):
        """Teste la classification en risque élevé."""
        result = predictor.predict(sample_data_from_csv)

        proba = result['probability_default']
        risk_level = result['risk_level']

        if proba >= 0.6:
            assert risk_level == 'high'

    def test_predict_preserves_client_id(self, predictor, sample_data_from_csv):
        """Teste que l'ID client est préservé."""
        if 'SK_ID_CURR' in sample_data_from_csv:
            original_id = sample_data_from_csv['SK_ID_CURR']
            result = predictor.predict(sample_data_from_csv)

            assert result['client_id'] == original_id


class TestPredictBatch:
    """Tests des prédictions en batch."""

    def test_predict_batch_returns_list(self, predictor, sample_data_from_csv):
        """Teste que predict_batch retourne une liste."""
        result = predictor.predict_batch([sample_data_from_csv])
        assert isinstance(result, list)

    def test_predict_batch_correct_length(self, predictor, sample_data_from_csv):
        """Teste que le nombre de prédictions correspond au nombre de clients."""
        n_clients = 3
        data_list = [sample_data_from_csv] * n_clients

        result = predictor.predict_batch(data_list)
        assert len(result) == n_clients

    def test_predict_batch_empty_list_raises_error(self, predictor):
        """Teste qu'une liste vide lève une erreur."""
        with pytest.raises(ValueError, match="vide"):
            predictor.predict_batch([])

    def test_predict_batch_too_large_raises_error(self, predictor, sample_data_from_csv):
        """Teste qu'un batch trop grand lève une erreur."""
        # Max 100 clients
        data_list = [sample_data_from_csv] * 101

        with pytest.raises(ValueError, match="Trop de clients"):
            predictor.predict_batch(data_list)

    def test_predict_batch_each_prediction_valid(self, predictor, sample_data_from_csv):
        """Teste que chaque prédiction du batch est valide."""
        n_clients = 5
        data_list = [sample_data_from_csv] * n_clients

        results = predictor.predict_batch(data_list)

        for result in results:
            assert 'probability_default' in result
            assert 'prediction' in result
            assert 'decision' in result
            assert 0.0 <= result['probability_default'] <= 1.0


class TestErrorHandling:
    """Tests de la gestion des erreurs."""

    def test_predict_with_missing_features_raises_error(self, predictor):
        """Teste qu'une prédiction avec features manquantes lève une erreur."""
        incomplete_data = {
            'SK_ID_CURR': 100001,
            'AMT_INCOME_TOTAL': 202500.0
            # Features manquantes...
        }

        with pytest.raises(ValueError, match="manquantes"):
            predictor.predict(incomplete_data)

    def test_predict_with_empty_dict_raises_error(self, predictor):
        """Teste qu'un dictionnaire vide lève une erreur."""
        with pytest.raises(Exception):
            predictor.predict({})


class TestModelInfo:
    """Tests de get_model_info."""

    def test_get_model_info_returns_dict(self, predictor):
        """Teste que get_model_info retourne un dictionnaire."""
        info = predictor.get_model_info()
        assert isinstance(info, dict)

    def test_get_model_info_has_required_fields(self, predictor):
        """Teste que get_model_info contient les champs requis."""
        info = predictor.get_model_info()

        required_fields = [
            'model_type',
            'model_version',
            'n_features',
            'threshold',
            'metrics'
        ]

        for field in required_fields:
            assert field in info

    def test_get_model_info_model_type_correct(self, predictor):
        """Teste que le type de modèle est correct."""
        info = predictor.get_model_info()
        assert info['model_type'] == 'LightGBM'

    def test_get_model_info_metrics_structure(self, predictor):
        """Teste que les métriques ont la bonne structure."""
        info = predictor.get_model_info()
        metrics = info['metrics']

        required_metrics = ['auc_roc', 'recall', 'precision', 'f1_score']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
