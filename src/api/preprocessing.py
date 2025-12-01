"""
Fonctions utilitaires pour le preprocessing des données.

Contient des fonctions auxiliaires pour le nettoyage et la validation
des données avant l'encodage.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def validate_data_types(data: Dict[str, Any], expected_types: Dict[str, type]) -> bool:
    """
    Valide que les types de données sont corrects.

    Args:
        data: Dictionnaire de données à valider
        expected_types: Dictionnaire des types attendus

    Returns:
        True si tous les types sont corrects

    Raises:
        ValueError: Si un type est incorrect
    """
    for key, expected_type in expected_types.items():
        if key in data:
            value = data[key]
            if value is not None and not isinstance(value, expected_type):
                raise ValueError(
                    f"Type incorrect pour {key}: attendu {expected_type.__name__}, "
                    f"reçu {type(value).__name__}"
                )
    return True


def handle_missing_values(df: pd.DataFrame, strategy: str = "default") -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans un DataFrame.

    Args:
        df: DataFrame à traiter
        strategy: Stratégie de gestion ("default", "drop", "fill")

    Returns:
        DataFrame avec les valeurs manquantes gérées
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy == "fill":
        # Remplir les valeurs numériques par la médiane
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # Remplir les valeurs catégorielles par le mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    return df


def validate_numerical_ranges(data: Dict[str, Any]) -> bool:
    """
    Valide que les valeurs numériques sont dans des plages raisonnables.

    Args:
        data: Dictionnaire de données à valider

    Returns:
        True si toutes les valeurs sont valides

    Raises:
        ValueError: Si une valeur est hors plage
    """
    # Définir les plages acceptables
    ranges = {
        'AMT_INCOME_TOTAL': (0, 1e9),
        'AMT_CREDIT': (0, 1e9),
        'AMT_ANNUITY': (0, 1e8),
        'AMT_GOODS_PRICE': (0, 1e9),
        'DAYS_BIRTH': (-25000, -6570),  # Entre 18 et 68 ans environ
        'DAYS_EMPLOYED': (-20000, 0),
    }

    for key, (min_val, max_val) in ranges.items():
        if key in data and data[key] is not None:
            value = data[key]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{key} hors plage: {value} (attendu entre {min_val} et {max_val})"
                )

    return True


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les noms de colonnes pour LightGBM.

    Args:
        df: DataFrame avec les colonnes à nettoyer

    Returns:
        DataFrame avec les noms de colonnes nettoyés
    """
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    return df


def check_feature_consistency(
    input_features: List[str],
    expected_features: List[str]
) -> Dict[str, List[str]]:
    """
    Vérifie la cohérence entre les features d'entrée et attendues.

    Args:
        input_features: Liste des features fournies
        expected_features: Liste des features attendues

    Returns:
        Dictionnaire avec les features manquantes et en trop
    """
    input_set = set(input_features)
    expected_set = set(expected_features)

    return {
        "missing": list(expected_set - input_set),
        "extra": list(input_set - expected_set)
    }
