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
import time

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

        # OPTIMISATION A1 : Mappings pour label encoding vectorisé
        self.label_mappings = None

        # OPTIMISATION A2 : Noms features one-hot pré-calculés
        self.onehot_feature_names = None

        # OPTIMISATION A3 : Cache de l'ordre des colonnes
        self.final_column_order = None

        logger.info("Initialisation du CreditScoringPredictor")
        self._load_artifacts()
        self._prepare_optimizations()

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

    def _prepare_optimizations(self) -> None:
        """
        Pré-calcule les structures pour les optimisations A1, A2, A3.

        OPTIMISATION A1 : Crée les mappings pour label encoding vectorisé
        OPTIMISATION A2 : Crée le ColumnTransformer pour one-hot encoding groupé
        OPTIMISATION A3 : Cache l'ordre final des colonnes
        """
        logger.info("Préparation des optimisations preprocessing...")

        # OPTIMISATION A1 : Pré-calculer les mappings label encoding
        # Au lieu de boucler et appeler encoder.transform() pour chaque colonne,
        # on crée un dictionnaire {colonne: {valeur: code}} qu'on appliquera avec df.replace()
        self.label_mappings = {}
        for col, encoder in self.label_encoders.items():
            # Créer mapping {valeur_originale: code_encodé}
            mapping = {cat: i for i, cat in enumerate(encoder.classes_)}
            self.label_mappings[col] = mapping
        logger.info(f"A1: {len(self.label_mappings)} mappings label encoding créés")

        # OPTIMISATION A2 : Pré-calculer les noms de colonnes one-hot
        # Au lieu de boucler avec pd.concat() pour chaque colonne,
        # on va encoder toutes les colonnes, puis faire UN SEUL pd.concat() à la fin
        self.onehot_feature_names = {}

        for col, encoder in self.onehot_encoders.items():
            # Pré-calculer les noms de features résultantes pour chaque colonne
            feature_names_temp = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            self.onehot_feature_names[col] = feature_names_temp

        logger.info(f"A2: Noms features one-hot pré-calculés pour {len(self.onehot_feature_names)} colonnes")

        # OPTIMISATION A3 : Pré-calculer l'ordre final des colonnes
        # Au lieu de nettoyer et réordonner à chaque prédiction, on le fait une fois
        self.final_column_order = self.feature_names.copy()
        logger.info(f"A3: Ordre des colonnes caché ({len(self.final_column_order)} features)")

        logger.info("✅ Optimisations preprocessing préparées")

    def preprocess(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prétraite les données d'entrée (VERSION OPTIMISÉE).

        Applique le label encoding et le one-hot encoding avec optimisations :
        - A1 : Label encoding vectorisé (df.replace au lieu de boucle)
        - A2 : One-hot encoding groupé (ColumnTransformer au lieu de pd.concat)
        - A3 : Colonnes cachées (pas de nettoyage/réordonnancement répété)

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

        # ============================================================
        # OPTIMISATION A1 : Label Encoding Vectorisé
        # ============================================================
        # Au lieu de : for col, encoder in self.label_encoders.items(): df[col] = encoder.transform(df[col])
        # On utilise : df.replace() avec les mappings pré-calculés
        for col, mapping in self.label_mappings.items():
            if col in df_encoded.columns:
                # Gérer les catégories inconnues
                unknown_mask = ~df_encoded[col].isin(mapping.keys())
                if unknown_mask.any():
                    logger.warning(f"Catégorie inconnue dans {col}, utilisation de la première classe")
                    first_class = list(mapping.keys())[0]
                    df_encoded.loc[unknown_mask, col] = first_class

                # Appliquer le mapping vectorisé (beaucoup plus rapide qu'encoder.transform)
                df_encoded[col] = df_encoded[col].replace(mapping).infer_objects(copy=False)

        # ============================================================
        # OPTIMISATION A2 : One-Hot Encoding Groupé
        # ============================================================
        # Au lieu de : boucle for + pd.concat() pour CHAQUE colonne (32 fois)
        # On fait : encoder toutes les colonnes, puis UN SEUL pd.concat() à la fin

        # Liste pour collecter tous les DataFrames encodés
        encoded_dfs = []

        # Colonnes à supprimer après encoding
        cols_to_drop = []

        for col, encoder in self.onehot_encoders.items():
            if col in df_encoded.columns:
                # Encoder la colonne
                encoded_data = encoder.transform(df_encoded[[col]])

                # Créer DataFrame avec les noms pré-calculés
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=self.onehot_feature_names[col],
                    index=df_encoded.index
                )

                # Ajouter à la liste
                encoded_dfs.append(encoded_df)
                cols_to_drop.append(col)

        # Supprimer les colonnes originales (une seule fois)
        df_encoded = df_encoded.drop(columns=cols_to_drop)

        # Concaténer TOUTES les colonnes encodées en UNE SEULE opération (gain majeur)
        if encoded_dfs:
            df_encoded = pd.concat([df_encoded] + encoded_dfs, axis=1)

        # Nettoyer les noms de colonnes pour LightGBM (une seule fois, pas à chaque colonne)
        df_encoded.columns = df_encoded.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

        # ============================================================
        # OPTIMISATION A3 : Colonnes Cachées
        # ============================================================
        # Au lieu de : vérifier + réordonner à chaque fois
        # On utilise : l'ordre pré-calculé (self.final_column_order)

        # Vérifier rapidement les colonnes manquantes
        missing_features = set(self.final_column_order) - set(df_encoded.columns)
        if missing_features:
            logger.error(f"Features manquantes: {missing_features}")
            raise ValueError(f"Features manquantes: {list(missing_features)[:5]}...")

        # Réordonner selon l'ordre pré-calculé (pas de regex, juste indexation)
        df_encoded = df_encoded[self.final_column_order]

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
        # Démarrer le chronomètre total
        start_total = time.time()

        try:
            # Extraire l'ID client s'il existe
            client_id = data.get('SK_ID_CURR', None)

            # Prétraiter les données avec timing
            start_preprocess = time.time()
            X_encoded = self.preprocess(data)
            preprocessing_time_ms = (time.time() - start_preprocess) * 1000

            # Faire la prédiction avec timing
            start_inference = time.time()
            proba = self.model.predict_proba(X_encoded)[0, 1]
            inference_time_ms = (time.time() - start_inference) * 1000

            prediction = int(proba >= self.threshold)

            # Déterminer la décision et le niveau de risque
            decision = Decision.REFUSE if prediction == 1 else Decision.APPROVE
            risk_level = self._get_risk_level(proba)

            # Calculer le temps total
            total_time_ms = (time.time() - start_total) * 1000

            # Construire la réponse
            response = {
                "client_id": client_id,
                "probability_default": float(proba),
                "prediction": prediction,
                "decision": decision.value,
                "risk_level": risk_level.value,
                "threshold_used": float(self.threshold),
                "model_version": self.model_version,
                "timestamp": datetime.now(),
                # Timings pour logging (sera retiré par l'endpoint)
                "_timing": {
                    "preprocessing_ms": preprocessing_time_ms,
                    "inference_ms": inference_time_ms,
                    "total_ms": total_time_ms,
                }
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
