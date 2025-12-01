"""
Tests des fonctions de validation des données.

Teste les fonctions du module preprocessing pour la validation
des types, des plages de valeurs, et de la cohérence des features.
"""

import pytest
import pandas as pd
import numpy as np

from src.api.preprocessing import (
    validate_data_types,
    validate_numerical_ranges,
    handle_missing_values,
    clean_column_names,
    check_feature_consistency
)


class TestValidateDataTypes:
    """Tests de la validation des types de données."""

    def test_validate_correct_types(self):
        """Teste la validation avec des types corrects."""
        data = {
            'AMT_INCOME_TOTAL': 150000.0,
            'AMT_CREDIT': 500000.0,
            'NAME_CONTRACT_TYPE': 'Cash loans'
        }

        expected_types = {
            'AMT_INCOME_TOTAL': (int, float),
            'AMT_CREDIT': (int, float),
            'NAME_CONTRACT_TYPE': str
        }

        # Doit passer sans erreur
        result = validate_data_types(data, expected_types)
        assert result is True

    def test_validate_incorrect_type_raises_error(self):
        """Teste qu'un type incorrect lève une erreur."""
        data = {
            'AMT_INCOME_TOTAL': "not_a_number"  # String au lieu de float
        }

        expected_types = {
            'AMT_INCOME_TOTAL': (int, float)
        }

        with pytest.raises(ValueError, match="Type incorrect"):
            validate_data_types(data, expected_types)

    def test_validate_missing_key_ignored(self):
        """Teste que les clés manquantes sont ignorées."""
        data = {
            'AMT_INCOME_TOTAL': 150000.0
            # AMT_CREDIT manquant
        }

        expected_types = {
            'AMT_INCOME_TOTAL': (int, float),
            'AMT_CREDIT': (int, float)
        }

        # Ne doit pas lever d'erreur si la clé n'est pas présente
        result = validate_data_types(data, expected_types)
        assert result is True

    def test_validate_none_values_ignored(self):
        """Teste que les valeurs None sont ignorées."""
        data = {
            'AMT_INCOME_TOTAL': None
        }

        expected_types = {
            'AMT_INCOME_TOTAL': (int, float)
        }

        # Ne doit pas lever d'erreur pour None
        result = validate_data_types(data, expected_types)
        assert result is True


class TestValidateNumericalRanges:
    """Tests de la validation des plages numériques."""

    def test_validate_amounts_in_range(self):
        """Teste la validation avec des montants valides."""
        data = {
            'AMT_INCOME_TOTAL': 150000.0,
            'AMT_CREDIT': 500000.0,
            'AMT_ANNUITY': 25000.0,
            'AMT_GOODS_PRICE': 450000.0
        }

        # Doit passer sans erreur
        result = validate_numerical_ranges(data)
        assert result is True

    def test_validate_negative_amount_raises_error(self):
        """Teste qu'un montant négatif lève une erreur."""
        data = {
            'AMT_INCOME_TOTAL': -1000.0  # Négatif invalide
        }

        with pytest.raises(ValueError, match="hors plage"):
            validate_numerical_ranges(data)

    def test_validate_amount_too_large_raises_error(self):
        """Teste qu'un montant trop élevé lève une erreur."""
        data = {
            'AMT_CREDIT': 1e10  # Trop élevé (max 1e9)
        }

        with pytest.raises(ValueError, match="hors plage"):
            validate_numerical_ranges(data)

    def test_validate_days_birth_valid(self):
        """Teste que DAYS_BIRTH dans une plage valide est accepté."""
        data = {
            'DAYS_BIRTH': -10000  # Environ 27 ans
        }

        # Doit passer sans erreur
        result = validate_numerical_ranges(data)
        assert result is True

    def test_validate_days_birth_too_young_raises_error(self):
        """Teste qu'un âge trop jeune lève une erreur."""
        data = {
            'DAYS_BIRTH': -5000  # Moins de 18 ans
        }

        with pytest.raises(ValueError, match="hors plage"):
            validate_numerical_ranges(data)

    def test_validate_days_birth_too_old_raises_error(self):
        """Teste qu'un âge trop élevé lève une erreur."""
        data = {
            'DAYS_BIRTH': -30000  # Plus de 68 ans environ
        }

        with pytest.raises(ValueError, match="hors plage"):
            validate_numerical_ranges(data)

    def test_validate_days_employed_valid(self):
        """Teste que DAYS_EMPLOYED dans une plage valide est accepté."""
        data = {
            'DAYS_EMPLOYED': -2000  # Environ 5 ans d'emploi
        }

        # Doit passer sans erreur
        result = validate_numerical_ranges(data)
        assert result is True

    def test_validate_days_employed_out_of_range_raises_error(self):
        """Teste que DAYS_EMPLOYED hors plage lève une erreur."""
        data = {
            'DAYS_EMPLOYED': -25000  # Trop ancien
        }

        with pytest.raises(ValueError, match="hors plage"):
            validate_numerical_ranges(data)

    def test_validate_days_employed_positive_raises_error(self):
        """Teste que DAYS_EMPLOYED positif lève une erreur."""
        data = {
            'DAYS_EMPLOYED': 1000  # Positif invalide
        }

        with pytest.raises(ValueError, match="hors plage"):
            validate_numerical_ranges(data)

    def test_validate_missing_keys_ignored(self):
        """Teste que les clés manquantes sont ignorées."""
        data = {
            'AMT_INCOME_TOTAL': 150000.0
            # Autres champs manquants
        }

        # Ne doit pas lever d'erreur
        result = validate_numerical_ranges(data)
        assert result is True

    def test_validate_none_values_ignored(self):
        """Teste que les valeurs None sont ignorées."""
        data = {
            'AMT_INCOME_TOTAL': None
        }

        # Ne doit pas lever d'erreur
        result = validate_numerical_ranges(data)
        assert result is True

    def test_validate_extreme_valid_values(self):
        """Teste les valeurs extrêmes mais valides."""
        data = {
            'AMT_INCOME_TOTAL': 999999999.0,  # Juste en dessous de 1e9
            'AMT_CREDIT': 1.0,  # Très petit mais > 0
            'DAYS_BIRTH': -6570,  # Juste 18 ans
            'DAYS_EMPLOYED': -19999  # Juste en dessous de -20000
        }

        # Doit passer sans erreur
        result = validate_numerical_ranges(data)
        assert result is True


class TestHandleMissingValues:
    """Tests de la gestion des valeurs manquantes."""

    def test_handle_missing_drop_strategy(self):
        """Teste la stratégie de suppression des valeurs manquantes."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8]
        })

        result = handle_missing_values(df, strategy='drop')

        # Doit supprimer les lignes avec NaN
        assert result.isna().sum().sum() == 0
        assert len(result) < len(df)

    def test_handle_missing_fill_strategy_numeric(self):
        """Teste le remplissage des valeurs numériques manquantes."""
        df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0]
        })

        result = handle_missing_values(df, strategy='fill')

        # Doit remplir par la médiane (3.0)
        assert result.isna().sum().sum() == 0
        assert result['numeric_col'].iloc[2] == 3.0

    def test_handle_missing_fill_strategy_categorical(self):
        """Teste le remplissage des valeurs catégorielles manquantes."""
        df = pd.DataFrame({
            'cat_col': ['A', 'A', None, 'B', 'A']
        })

        result = handle_missing_values(df, strategy='fill')

        # Doit remplir par le mode ('A')
        assert result.isna().sum().sum() == 0
        assert result['cat_col'].iloc[2] == 'A'

    def test_handle_missing_default_strategy(self):
        """Teste la stratégie par défaut (ne rien faire)."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4]
        })

        result = handle_missing_values(df, strategy='default')

        # Les NaN doivent rester
        assert result.isna().sum().sum() == 1


class TestCleanColumnNames:
    """Tests du nettoyage des noms de colonnes."""

    def test_clean_special_characters(self):
        """Teste le nettoyage des caractères spéciaux."""
        df = pd.DataFrame({
            'col-name': [1],
            'col name': [2],
            'col@name': [3],
            'col#name': [4]
        })

        result = clean_column_names(df)

        # Tous les caractères spéciaux doivent être remplacés par _
        for col in result.columns:
            assert '-' not in col
            assert ' ' not in col
            assert '@' not in col
            assert '#' not in col

    def test_clean_preserves_valid_names(self):
        """Teste que les noms valides sont préservés."""
        df = pd.DataFrame({
            'valid_name': [1],
            'AnotherValid123': [2],
            'column_123_abc': [3]
        })

        result = clean_column_names(df)

        # Les noms valides doivent rester inchangés
        assert 'valid_name' in result.columns
        assert 'AnotherValid123' in result.columns
        assert 'column_123_abc' in result.columns

    def test_clean_replaces_multiple_special_chars(self):
        """Teste le remplacement de plusieurs caractères spéciaux."""
        df = pd.DataFrame({
            'col-@-name': [1]
        })

        result = clean_column_names(df)

        # Les caractères spéciaux doivent être remplacés par _
        assert result.columns[0] == 'col___name'


class TestCheckFeatureConsistency:
    """Tests de la vérification de cohérence des features."""

    def test_check_all_features_present(self):
        """Teste quand toutes les features sont présentes."""
        input_features = ['A', 'B', 'C']
        expected_features = ['A', 'B', 'C']

        result = check_feature_consistency(input_features, expected_features)

        assert len(result['missing']) == 0
        assert len(result['extra']) == 0

    def test_check_missing_features(self):
        """Teste la détection de features manquantes."""
        input_features = ['A', 'B']
        expected_features = ['A', 'B', 'C', 'D']

        result = check_feature_consistency(input_features, expected_features)

        assert len(result['missing']) == 2
        assert 'C' in result['missing']
        assert 'D' in result['missing']
        assert len(result['extra']) == 0

    def test_check_extra_features(self):
        """Teste la détection de features en trop."""
        input_features = ['A', 'B', 'C', 'D']
        expected_features = ['A', 'B']

        result = check_feature_consistency(input_features, expected_features)

        assert len(result['missing']) == 0
        assert len(result['extra']) == 2
        assert 'C' in result['extra']
        assert 'D' in result['extra']

    def test_check_both_missing_and_extra(self):
        """Teste avec des features manquantes et en trop."""
        input_features = ['A', 'B', 'E', 'F']
        expected_features = ['A', 'B', 'C', 'D']

        result = check_feature_consistency(input_features, expected_features)

        assert len(result['missing']) == 2
        assert 'C' in result['missing']
        assert 'D' in result['missing']

        assert len(result['extra']) == 2
        assert 'E' in result['extra']
        assert 'F' in result['extra']

    def test_check_empty_input(self):
        """Teste avec une liste d'entrée vide."""
        input_features = []
        expected_features = ['A', 'B', 'C']

        result = check_feature_consistency(input_features, expected_features)

        assert len(result['missing']) == 3
        assert len(result['extra']) == 0

    def test_check_empty_expected(self):
        """Teste avec une liste attendue vide."""
        input_features = ['A', 'B', 'C']
        expected_features = []

        result = check_feature_consistency(input_features, expected_features)

        assert len(result['missing']) == 0
        assert len(result['extra']) == 3

    def test_check_both_empty(self):
        """Teste avec les deux listes vides."""
        input_features = []
        expected_features = []

        result = check_feature_consistency(input_features, expected_features)

        assert len(result['missing']) == 0
        assert len(result['extra']) == 0


class TestIntegrationValidation:
    """Tests d'intégration pour la validation."""

    def test_complete_validation_pipeline_valid_data(self):
        """Teste un pipeline complet avec des données valides."""
        data = {
            'AMT_INCOME_TOTAL': 150000.0,
            'AMT_CREDIT': 500000.0,
            'AMT_ANNUITY': 25000.0,
            'DAYS_BIRTH': -10000
        }

        expected_types = {
            'AMT_INCOME_TOTAL': (int, float),
            'AMT_CREDIT': (int, float),
            'AMT_ANNUITY': (int, float),
            'DAYS_BIRTH': (int, float)
        }

        # Toutes les validations doivent passer
        assert validate_data_types(data, expected_types)
        assert validate_numerical_ranges(data)

    def test_complete_validation_pipeline_invalid_data(self):
        """Teste un pipeline complet avec des données invalides."""
        data = {
            'AMT_INCOME_TOTAL': -1000.0,  # Négatif invalide
            'AMT_CREDIT': "not_a_number"  # Type incorrect
        }

        expected_types = {
            'AMT_INCOME_TOTAL': (int, float),
            'AMT_CREDIT': (int, float)
        }

        # Validation des types doit échouer
        with pytest.raises(ValueError):
            validate_data_types(data, expected_types)

        # Validation des plages doit échouer
        data['AMT_CREDIT'] = 500000.0  # Corriger le type
        with pytest.raises(ValueError):
            validate_numerical_ranges(data)
