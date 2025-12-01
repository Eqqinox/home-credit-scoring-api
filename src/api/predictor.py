"""
Classe Predictor pour le chargement du modèle et les prédictions.

Charge tous les artefacts au démarrage et fournit des méthodes
pour prétraiter les données et faire des prédictions.
"""

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

from .config import settings
from .schemas import RiskLevel, Decision

logger = logging.getLogger(__name__)


class CreditScoringPredictor:
    """
    Predictor pour le scoring crédit.

    Charge le modèle et les encoders une seule fois au démarrage,
    puis les réutilise pour toutes les prédictions.
    """

    def __init__(self):
        """Initialise le predictor en chargeant tous les artefacts."""
        self.model = None
        self.label_encoders = None
        self.onehot_encoders = None
        self.feature_names = None
        self.threshold = None
        self.metrics = None
        self.model_version = settings.api_version

        logger.info("Initialisation du CreditScoringPredictor")
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Charge tous les artefacts du modèle.

        Raises:
            FileNotFoundError: Si un artefact est manquant
            Exception: Si le chargement échoue
        """
        try:
            # 1. Charger le modèle
            logger.info(f"Chargement du modèle depuis {settings.model_path}")
            with open(settings.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Modèle chargé: {type(self.model).__name__}")

            # 2. Charger les label encoders
            logger.info(f"Chargement des label encoders depuis {settings.label_encoders_path}")
            with open(settings.label_encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            logger.info(f"{len(self.label_encoders)} label encoders chargés")

            # 3. Charger les one-hot encoders
            logger.info(f"Chargement des one-hot encoders depuis {settings.onehot_encoder_path}")
            with open(settings.onehot_encoder_path, 'rb') as f:
                self.onehot_encoders = pickle.load(f)
            logger.info(f"{len(self.onehot_encoders)} one-hot encoders chargés")

            # 4. Charger les noms de features
            logger.info(f"Chargement des noms de features depuis {settings.feature_names_path}")
            with open(settings.feature_names_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            logger.info(f"{len(self.feature_names)} features attendues")

            # 5. Charger le seuil optimal
            logger.info(f"Chargement du seuil depuis {settings.threshold_path}")
            with open(settings.threshold_path, 'r') as f:
                threshold_data = json.load(f)
            self.threshold = threshold_data['optimal_threshold']
            logger.info(f"Seuil optimal: {self.threshold:.4f}")

            # 6. Charger les métriques
            logger.info(f"Chargement des métriques depuis {settings.metrics_path}")
            with open(settings.metrics_path, 'r') as f:
                self.metrics = json.load(f)
            logger.info(f"Métriques chargées: AUC-ROC={self.metrics['auc_roc']:.4f}")

            logger.info("Tous les artefacts ont été chargés avec succès")

        except FileNotFoundError as e:
            logger.error(f"Artefact manquant: {e}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du chargement des artefacts: {e}")
            raise

    def preprocess(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prétraite les données d'entrée.

        Applique le label encoding et le one-hot encoding comme lors
        de l'entraînement.

        Args:
            data: Dictionnaire contenant les features du client

        Returns:
            DataFrame avec les features encodées et prêtes pour le modèle

        Raises:
            ValueError: Si des features requises sont manquantes
        """
        # Convertir en DataFrame
        df = pd.DataFrame([data])

        # Retirer SK_ID_CURR si présent (pas utilisé par le modèle)
        if 'SK_ID_CURR' in df.columns:
            df = df.drop(columns=['SK_ID_CURR'])

        # Copier pour le preprocessing
        df_encoded = df.copy()

        # Label Encoding
        for col, encoder in self.label_encoders.items():
            if col in df_encoded.columns:
                # Gérer les catégories inconnues
                unknown_mask = ~df_encoded[col].isin(encoder.classes_)
                if unknown_mask.any():
                    logger.warning(f"Catégorie inconnue dans {col}, utilisation de la première classe")
                    df_encoded.loc[unknown_mask, col] = encoder.classes_[0]
                df_encoded[col] = encoder.transform(df_encoded[col])

        # One-Hot Encoding
        for col, encoder in self.onehot_encoders.items():
            if col in df_encoded.columns:
                # Encoder
                encoded_data = encoder.transform(df_encoded[[col]])
                feature_names_temp = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=feature_names_temp,
                    index=df_encoded.index
                )
                # Remplacer la colonne originale par les colonnes encodées
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

        # Nettoyer les noms de colonnes pour LightGBM
        df_encoded.columns = df_encoded.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

        # Vérifier que toutes les features attendues sont présentes
        missing_features = set(self.feature_names) - set(df_encoded.columns)
        if missing_features:
            logger.error(f"Features manquantes: {missing_features}")
            raise ValueError(f"Features manquantes: {list(missing_features)[:5]}...")

        # Réordonner les colonnes selon l'ordre attendu
        df_encoded = df_encoded[self.feature_names]

        return df_encoded

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fait une prédiction pour un client.

        Args:
            data: Dictionnaire contenant les features du client

        Returns:
            Dictionnaire avec la prédiction et les métadonnées

        Raises:
            ValueError: Si les données sont invalides
            Exception: Si la prédiction échoue
        """
        try:
            # Extraire l'ID client s'il existe
            client_id = data.get('SK_ID_CURR', None)

            # Prétraiter les données
            X_encoded = self.preprocess(data)

            # Faire la prédiction
            proba = self.model.predict_proba(X_encoded)[0, 1]
            prediction = int(proba >= self.threshold)

            # Déterminer la décision et le niveau de risque
            decision = Decision.REFUSE if prediction == 1 else Decision.APPROVE
            risk_level = self._get_risk_level(proba)

            # Construire la réponse
            response = {
                "client_id": client_id,
                "probability_default": float(proba),
                "prediction": prediction,
                "decision": decision.value,
                "risk_level": risk_level.value,
                "threshold_used": float(self.threshold),
                "model_version": self.model_version,
                "timestamp": datetime.now()
            }

            logger.info(f"Prédiction réussie pour client {client_id}: proba={proba:.4f}, decision={decision.value}")
            return response

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise

    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fait des prédictions pour plusieurs clients.

        Args:
            data_list: Liste de dictionnaires contenant les features des clients

        Returns:
            Liste de dictionnaires avec les prédictions

        Raises:
            ValueError: Si la liste est vide ou trop grande
        """
        if not data_list:
            raise ValueError("La liste de données est vide")

        if len(data_list) > settings.max_batch_size:
            raise ValueError(f"Trop de clients (max: {settings.max_batch_size})")

        predictions = []
        for data in data_list:
            try:
                pred = self.predict(data)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Erreur pour client {data.get('SK_ID_CURR')}: {e}")
                # On peut choisir de skip ou de lever l'erreur
                raise

        return predictions

    def _get_risk_level(self, probability: float) -> RiskLevel:
        """
        Détermine le niveau de risque basé sur la probabilité.

        Args:
            probability: Probabilité de défaut (0-1)

        Returns:
            Niveau de risque (LOW, MEDIUM, HIGH)
        """
        if probability < 0.3:
            return RiskLevel.LOW
        elif probability < 0.6:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le modèle.

        Returns:
            Dictionnaire avec les informations du modèle
        """
        return {
            "model_type": "LightGBM",
            "model_version": self.model_version,
            "n_features": len(self.feature_names),
            "threshold": self.threshold,
            "metrics": {
                "auc_roc": self.metrics['auc_roc'],
                "recall": self.metrics['recall'],
                "precision": self.metrics['precision'],
                "f1_score": self.metrics['f1_score'],
                "business_cost": self.metrics['business_cost']['total_cost']
            },
            "training_date": "2025-12-01"  # À paramétrer
        }

    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé."""
        return self.model is not None
