"""
Simulation de trafic API pour le scoring crédit.

Génère des requêtes de prédiction en streaming avec variations
pour simuler du data drift et alimenter PostgreSQL.

Usage:
    python src/scripts/simulate_traffic.py [--num-predictions 100] [--delay 0.5]

Requirements:
    - API lancée sur http://localhost:8000
    - PostgreSQL opérationnel
    - Dataset: data/reference/train_reference.parquet
"""

import pandas as pd
import requests
import time
import random
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import json


class TrafficSimulator:
    """
    Simulateur de trafic pour l'API de scoring crédit.

    Génère des requêtes de prédiction en streaming avec variations
    pour simuler du data drift.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        dataset_path: str = "data/reference/train_reference.parquet",
        num_predictions: int = 100,
        delay_seconds: float = 0.5,
        drift_probability: float = 0.3,
        drift_magnitude: float = 0.15
    ):
        """
        Initialise le simulateur de trafic.

        Args:
            api_url: URL de l'API de scoring
            dataset_path: Chemin vers le dataset de référence (Parquet)
            num_predictions: Nombre de prédictions à générer
            delay_seconds: Délai entre chaque requête (simulation streaming)
            drift_probability: Probabilité d'appliquer des variations (0-1)
            drift_magnitude: Magnitude des variations (0-1, ±15% par défaut)
        """
        self.api_url = api_url
        self.dataset_path = dataset_path
        self.num_predictions = num_predictions
        self.delay_seconds = delay_seconds
        self.drift_probability = drift_probability
        self.drift_magnitude = drift_magnitude

        # Statistiques de la simulation
        self.stats = {
            'total': 0,
            'success': 0,
            'failures': 0,
            'approvals': 0,
            'refusals': 0,
            'response_times': [],
            'drifted_clients': 0
        }

        # Dataset
        self.df: Optional[pd.DataFrame] = None
        self.available_indices: List[int] = []

    @staticmethod
    def prepare_reference_dataset(
        csv_path: str = "data/app_train_models.csv",
        output_path: str = "data/reference/train_reference.parquet"
    ) -> None:
        """
        Convertit le CSV en Parquet (une seule fois).

        Args:
            csv_path: Chemin vers le fichier CSV source
            output_path: Chemin de sortie pour le fichier Parquet
        """
        output_file = Path(output_path)

        # Vérifier si le fichier parquet existe déjà
        if output_file.exists():
            print(f"✅ Dataset de référence déjà existant : {output_path}")
            return

        print(f"📦 Conversion {csv_path} → {output_path}...")

        # Créer le répertoire si nécessaire
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Charger et convertir
        df = pd.read_csv(csv_path)

        # Supprimer la colonne TARGET (pas nécessaire pour la simulation)
        if 'TARGET' in df.columns:
            df = df.drop(columns=['TARGET'])

        # Sauvegarder en Parquet (compression gzip)
        df.to_parquet(output_path, compression='gzip', index=False)

        print(f"✅ Conversion terminée : {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"   Taille CSV    : {Path(csv_path).stat().st_size / 1024**2:.1f} MiB")
        print(f"   Taille Parquet: {output_file.stat().st_size / 1024**2:.1f} MiB")

    def load_dataset(self) -> pd.DataFrame:
        """
        Charge le dataset de référence depuis le fichier Parquet.

        Returns:
            DataFrame prêt pour l'API

        Raises:
            FileNotFoundError: Si le dataset n'existe pas
        """
        dataset_file = Path(self.dataset_path)

        if not dataset_file.exists():
            # Essayer de le créer depuis le CSV
            csv_path = Path("data/app_train_models.csv")
            if csv_path.exists():
                print(f"⚠️  Dataset Parquet manquant, conversion depuis CSV...")
                self.prepare_reference_dataset()
            else:
                raise FileNotFoundError(
                    f"Dataset introuvable : {self.dataset_path}\n"
                    f"CSV source également introuvable : {csv_path}"
                )

        # Charger le Parquet
        self.df = pd.read_parquet(self.dataset_path)

        # Initialiser les indices disponibles
        self.available_indices = list(range(len(self.df)))

        return self.df

    def select_random_clients(self, n: int) -> List[Dict[str, Any]]:
        """
        Sélectionne n clients aléatoires du dataset.

        Args:
            n: Nombre de clients à sélectionner

        Returns:
            Liste de dictionnaires (JSON-ready)
        """
        if not self.available_indices:
            # Réinitialiser si on a épuisé les clients
            self.available_indices = list(range(len(self.df)))

        # Sélectionner n indices aléatoires
        selected_indices = random.sample(
            self.available_indices,
            min(n, len(self.available_indices))
        )

        # Retirer ces indices pour éviter les doublons
        for idx in selected_indices:
            self.available_indices.remove(idx)

        # Convertir en dictionnaires
        clients = []
        for idx in selected_indices:
            client_data = self.df.iloc[idx].to_dict()

            # Convertir les NaN en None pour JSON
            client_data = {
                k: (None if pd.isna(v) else v)
                for k, v in client_data.items()
            }

            clients.append(client_data)

        return clients

    def apply_drift_variations(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applique des variations pour simuler du data drift.

        Args:
            client_data: Données originales du client

        Returns:
            Données modifiées avec drift
        """
        drifted = client_data.copy()

        # Features numériques à faire varier (revenus, crédits, mensualités)
        numeric_features = [
            'AMT_INCOME_TOTAL',
            'AMT_CREDIT',
            'AMT_ANNUITY',
            'AMT_GOODS_PRICE'
        ]

        for feature in numeric_features:
            if feature in drifted and drifted[feature] is not None:
                # Variation : ±drift_magnitude
                variation = random.uniform(
                    1 - self.drift_magnitude,
                    1 + self.drift_magnitude
                )
                drifted[feature] = float(drifted[feature]) * variation

        # EXT_SOURCE : Légère dégradation pour simuler du drift
        ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        for feature in ext_sources:
            if feature in drifted and drifted[feature] is not None:
                # Variation plus faible : ±10%
                variation = random.uniform(0.9, 1.1)
                drifted[feature] = float(drifted[feature]) * variation
                # Garder dans [0, 1]
                drifted[feature] = max(0.0, min(1.0, drifted[feature]))

        # DAYS_EMPLOYED : Augmentation progressive (clients vieillissent)
        if 'DAYS_EMPLOYED' in drifted and drifted['DAYS_EMPLOYED'] is not None:
            # Ajouter entre 0 et 365 jours d'ancienneté (valeurs négatives)
            drifted['DAYS_EMPLOYED'] = float(drifted['DAYS_EMPLOYED']) - random.randint(0, 365)

        return drifted

    def send_prediction_request(
        self,
        client_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], float]:
        """
        Envoie une requête de prédiction à l'API.

        Args:
            client_data: Données du client

        Returns:
            (success, response_data, elapsed_time_ms)
        """
        url = f"{self.api_url}/predict"
        headers = {"Content-Type": "application/json"}

        try:
            start = time.time()
            response = requests.post(
                url,
                json=client_data,
                headers=headers,
                timeout=10
            )
            elapsed_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                return True, response.json(), elapsed_ms
            else:
                return False, {
                    'error': f"HTTP {response.status_code}",
                    'detail': response.text[:200]
                }, elapsed_ms

        except requests.exceptions.Timeout:
            return False, {'error': 'Timeout (>10s)'}, 10000.0
        except requests.exceptions.ConnectionError:
            return False, {'error': 'Connection refused'}, 0.0
        except Exception as e:
            return False, {'error': str(e)}, 0.0

    def verify_api_health(self) -> bool:
        """
        Vérifie que l'API est accessible et opérationnelle.

        Returns:
            True si l'API répond, sinon lève une exception

        Raises:
            ConnectionError: Si l'API n'est pas accessible
        """
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API opérationnelle (version {data.get('model_version', 'inconnue')})")
                return True
            else:
                raise ConnectionError(f"API retourne HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"L'API n'est pas accessible sur {self.api_url}\n"
                f"Lancez l'API avec : uvicorn src.api.main:app --reload"
            )
        except Exception as e:
            raise ConnectionError(f"Erreur lors de la vérification de l'API : {e}")

    def _log_result(
        self,
        request_num: int,
        client_data: Dict[str, Any],
        response: Dict[str, Any],
        success: bool,
        elapsed_ms: float
    ) -> None:
        """
        Log le résultat d'une prédiction.

        Args:
            request_num: Numéro de la requête
            client_data: Données envoyées
            response: Réponse de l'API
            success: Succès ou échec
            elapsed_ms: Temps de réponse en ms
        """
        if success:
            client_id = response.get('client_id', client_data.get('SK_ID_CURR', 'N/A'))
            decision = response.get('decision', 'unknown')
            proba = response.get('probability_default', 0.0)

            # Mettre à jour les stats
            self.stats['approvals'] += 1 if decision == 'approve' else 0
            self.stats['refusals'] += 1 if decision == 'refuse' else 0

            # Log succinct (on a déjà la barre de progression)
            # print(f"[{request_num:3d}] Client {client_id} → {decision.upper():7s} (proba={proba:.3f}, {elapsed_ms:.1f}ms)")
        else:
            error_msg = response.get('error', 'Unknown error')
            print(f"\n❌ [{request_num:3d}] ERREUR : {error_msg}")

    def run_simulation(self) -> None:
        """
        Lance la simulation complète.
        """
        print(f"\n🚀 Démarrage de la simulation ({self.num_predictions} prédictions)...")

        # Barre de progression
        with tqdm(total=self.num_predictions, desc="Simulation", unit="req") as pbar:
            for i in range(self.num_predictions):
                # 1. Sélectionner un client aléatoire
                client = self.select_random_clients(1)[0]

                # 2. Appliquer drift (probabilité drift_probability)
                drift_applied = False
                if random.random() < self.drift_probability:
                    client = self.apply_drift_variations(client)
                    drift_applied = True
                    self.stats['drifted_clients'] += 1

                # 3. Envoyer la requête
                success, response, elapsed_ms = self.send_prediction_request(client)

                # 4. Mettre à jour les stats
                self.stats['total'] += 1
                if success:
                    self.stats['success'] += 1
                    self.stats['response_times'].append(elapsed_ms)
                else:
                    self.stats['failures'] += 1

                # 5. Logger le résultat
                self._log_result(i + 1, client, response, success, elapsed_ms)

                # 6. Attendre avant la prochaine requête (streaming)
                if i < self.num_predictions - 1:  # Pas de délai après la dernière
                    time.sleep(self.delay_seconds)

                # 7. Mettre à jour la barre de progression
                pbar.update(1)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de la simulation.

        Returns:
            Dictionnaire de statistiques
        """
        success_rate = (self.stats['success'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        approval_rate = (self.stats['approvals'] / self.stats['success'] * 100) if self.stats['success'] > 0 else 0
        refusal_rate = (self.stats['refusals'] / self.stats['success'] * 100) if self.stats['success'] > 0 else 0

        avg_response_time = (
            sum(self.stats['response_times']) / len(self.stats['response_times'])
            if self.stats['response_times'] else 0
        )

        return {
            'total': self.stats['total'],
            'success': self.stats['success'],
            'failures': self.stats['failures'],
            'success_rate': success_rate,
            'approvals': self.stats['approvals'],
            'refusals': self.stats['refusals'],
            'approval_rate': approval_rate,
            'refusal_rate': refusal_rate,
            'avg_response_time': avg_response_time,
            'drifted_clients': self.stats['drifted_clients']
        }


def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(
        description="Simulateur de trafic pour l'API de scoring crédit"
    )
    parser.add_argument(
        "--num-predictions",
        type=int,
        default=100,
        help="Nombre de prédictions à générer (défaut: 100)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Délai entre requêtes en secondes (défaut: 0.5)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="URL de l'API (défaut: http://localhost:8000)"
    )
    parser.add_argument(
        "--drift-prob",
        type=float,
        default=0.3,
        help="Probabilité de drift (0-1, défaut: 0.3)"
    )
    parser.add_argument(
        "--drift-mag",
        type=float,
        default=0.15,
        help="Magnitude du drift (0-1, défaut: 0.15)"
    )

    args = parser.parse_args()

    # Bannière
    print("=" * 80)
    print("SIMULATION DE TRAFIC - HOME CREDIT SCORING API")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Nombre de prédictions : {args.num_predictions}")
    print(f"  - Délai entre requêtes  : {args.delay}s")
    print(f"  - Probabilité de drift  : {args.drift_prob * 100}%")
    print(f"  - Magnitude du drift    : ±{args.drift_mag * 100}%")
    print("=" * 80)

    # Initialiser le simulateur
    simulator = TrafficSimulator(
        api_url=args.api_url,
        num_predictions=args.num_predictions,
        delay_seconds=args.delay,
        drift_probability=args.drift_prob,
        drift_magnitude=args.drift_mag
    )

    try:
        # 1. Vérifier que l'API est accessible
        print("\n[1/3] Vérification de l'API...")
        simulator.verify_api_health()

        # 2. Charger le dataset
        print("\n[2/3] Chargement du dataset...")
        simulator.load_dataset()
        print(f"✅ {len(simulator.df)} clients disponibles")

        # 3. Lancer la simulation
        print("\n[3/3] Lancement de la simulation...")
        start_time = time.time()
        simulator.run_simulation()
        elapsed = time.time() - start_time

        # 4. Afficher les statistiques
        print("\n" + "=" * 80)
        print("RÉSULTATS DE LA SIMULATION")
        print("=" * 80)
        stats = simulator.get_statistics()
        print(f"Durée totale          : {elapsed:.2f}s")
        print(f"Requêtes envoyées     : {stats['total']}")
        print(f"Succès                : {stats['success']} ({stats['success_rate']:.1f}%)")
        print(f"Échecs                : {stats['failures']}")
        print(f"Approbations          : {stats['approvals']} ({stats['approval_rate']:.1f}%)")
        print(f"Refus                 : {stats['refusals']} ({stats['refusal_rate']:.1f}%)")
        print(f"Temps moyen           : {stats['avg_response_time']:.2f}ms")
        print(f"Clients avec drift    : {stats['drifted_clients']}")
        print("=" * 80)

        print("\n✅ Simulation terminée avec succès !")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        print("\nVérifications :")
        print("  1. L'API est-elle lancée ? → uvicorn src.api.main:app --reload")
        print("  2. PostgreSQL est-il actif ? → brew services list")
        print("  3. Le dataset existe-t-il ? → ls -lh data/")
        sys.exit(1)


if __name__ == "__main__":
    main()
