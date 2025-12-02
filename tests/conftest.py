"""
Fixtures partagées pour les tests.
Contient les fixtures communes utilisées par plusieurs modules de test.
"""

import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.predictor import CreditScoringPredictor


@pytest.fixture
def api_client():
    """
    Fixture fournissant un client de test pour l'API.
    
    Returns:
        TestClient FastAPI pour tester les endpoints
    """
    # Créer d'abord le TestClient (cela déclenche le lifespan et charge le modèle)
    client = TestClient(app)
    
    # Ensuite importer et vérifier le predictor
    from src.api.main import predictor
    
    # Vérifier que le modèle est chargé
    if predictor is None or not predictor.is_loaded():
        pytest.skip("Le modèle n'a pas pu être chargé pour les tests")
    
    return client


@pytest.fixture
def valid_client_data():
    """
    Fixture fournissant des données client valides.
    
    Returns:
        Dictionnaire avec des données client complètes et valides
    """
    return {
        "SK_ID_CURR": 100001,
        "AMT_INCOME_TOTAL": 202500.0,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "AMT_GOODS_PRICE": 351000.0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        # Ajouter toutes les autres features nécessaires...
        # Pour les tests, on peut charger un exemple réel
    }


@pytest.fixture
def invalid_client_data():
    """
    Fixture fournissant des données client invalides.
    
    Returns:
        Dictionnaire avec des données incorrectes
    """
    return {
        "SK_ID_CURR": 100002,
        "AMT_INCOME_TOTAL": -1000.0,  # Montant négatif invalide
        "AMT_CREDIT": 1e15,  # Montant trop élevé
        "NAME_CONTRACT_TYPE": "Invalid Type",
    }


@pytest.fixture
def client_data_missing_features():
    """
    Fixture fournissant des données avec des features manquantes.
    
    Returns:
        Dictionnaire incomplet
    """
    return {
        "SK_ID_CURR": 100003,
        "AMT_INCOME_TOTAL": 150000.0,
        # Features manquantes...
    }


@pytest.fixture
def sample_data_from_csv():
    """
    Charge un échantillon réel depuis le CSV.
    
    Returns:
        Dictionnaire avec des données réelles
    """
    import pandas as pd
    
    data_path = Path("data/app_train_models.csv")
    
    if data_path.exists():
        df = pd.read_csv(data_path, nrows=1)
        # Convertir en dictionnaire sans la target
        data = df.drop(columns=['TARGET'], errors='ignore').iloc[0].to_dict()
        return data
    else:
        pytest.skip("Fichier de données non trouvé")


@pytest.fixture
def predictor():
    """
    Fixture fournissant une instance du predictor.
    
    Returns:
        Instance de CreditScoringPredictor
    """
    return CreditScoringPredictor()


@pytest.fixture
def expected_features():
    """
    Fixture fournissant la liste des features attendues.
    
    Returns:
        Liste des noms de features
    """
    import pickle
    feature_path = Path("models/feature_names.pkl")
    
    if feature_path.exists():
        with open(feature_path, 'rb') as f:
            return pickle.load(f)
    else:
        pytest.skip("Fichier de features non trouvé")


@pytest.fixture
def model_metrics():
    """
    Fixture fournissant les métriques du modèle.
    
    Returns:
        Dictionnaire des métriques
    """
    metrics_path = Path("models/metrics.json")
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    else:
        pytest.skip("Fichier de métriques non trouvé")