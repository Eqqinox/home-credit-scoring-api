"""
Script de profiling pour analyser les performances de l'API avec cProfile.

Ce script utilise cProfile pour identifier les fonctions les plus coûteuses
dans le preprocessing et l'inférence du modèle. Il génère des rapports .prof
et analyse les résultats.

Usage:
    python src/scripts/profile_api.py

Output:
    - reports/profiling/baseline_single.prof
    - reports/profiling/baseline_batch.prof
    - Affichage console du top 20 fonctions lentes
"""

import sys
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, Any, List
import random
import json

# Ajouter le répertoire src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.predictor import CreditScoringPredictor


def load_sample_data() -> Dict[str, Any]:
    """
    Charge un exemple de données pour le profiling.

    Returns:
        Dictionnaire représentant les features d'un client.
    """
    # Charger l'exemple depuis example_single_request.json
    example_path = Path("example_single_request.json")

    if example_path.exists():
        with open(example_path, 'r') as f:
            data = json.load(f)
            return data
    else:
        # Fallback : créer des données de test minimales
        print("⚠️  example_single_request.json non trouvé, utilisation de données de test")
        return {
            "SK_ID_CURR": 100001,
            "AMT_CREDIT": 406597.5,
            "AMT_INCOME_TOTAL": 202500.0,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "FLAG_OWN_CAR": "N"
        }


def generate_batch_data(n: int = 100) -> List[Dict[str, Any]]:
    """
    Génère un batch de données pour le profiling.

    Args:
        n: Nombre de prédictions dans le batch.

    Returns:
        Liste de dictionnaires de features.
    """
    base_data = load_sample_data()
    batch = []

    for i in range(n):
        # Créer une variation des données de base
        data_copy = base_data.copy()
        # Varier légèrement les valeurs numériques
        if "AMT_CREDIT" in data_copy:
            data_copy["AMT_CREDIT"] *= random.uniform(0.8, 1.2)
        if "AMT_INCOME_TOTAL" in data_copy:
            data_copy["AMT_INCOME_TOTAL"] *= random.uniform(0.8, 1.2)

        batch.append(data_copy)

    return batch


def profile_single_predictions(predictor: CreditScoringPredictor, n: int = 1000) -> cProfile.Profile:
    """
    Profile n prédictions individuelles.

    Args:
        predictor: Instance du prédicteur.
        n: Nombre de prédictions à exécuter.

    Returns:
        Objet Profile de cProfile.
    """
    print(f"\n🔍 Profiling de {n} prédictions individuelles...")

    sample_data = load_sample_data()

    profiler = cProfile.Profile()
    profiler.enable()

    # Exécuter n prédictions
    for i in range(n):
        predictor.predict(sample_data)
        if (i + 1) % 100 == 0:
            print(f"   Progression : {i + 1}/{n} prédictions")

    profiler.disable()

    print(f"✅ Profiling single terminé ({n} prédictions)")
    return profiler


def profile_batch_predictions(predictor: CreditScoringPredictor, n_batches: int = 10, batch_size: int = 100) -> cProfile.Profile:
    """
    Profile n_batches batches de prédictions.

    Args:
        predictor: Instance du prédicteur.
        n_batches: Nombre de batches à exécuter.
        batch_size: Taille de chaque batch.

    Returns:
        Objet Profile de cProfile.
    """
    print(f"\n🔍 Profiling de {n_batches} batches de {batch_size} prédictions...")

    profiler = cProfile.Profile()
    profiler.enable()

    # Exécuter n_batches batches
    for i in range(n_batches):
        batch_data = generate_batch_data(batch_size)
        predictor.predict_batch(batch_data)
        print(f"   Batch {i + 1}/{n_batches} terminé")

    profiler.disable()

    print(f"✅ Profiling batch terminé ({n_batches} batches × {batch_size} prédictions)")
    return profiler


def save_profile(profiler: cProfile.Profile, output_path: Path) -> None:
    """
    Sauvegarde le profil dans un fichier .prof.

    Args:
        profiler: Objet Profile de cProfile.
        output_path: Chemin du fichier de sortie.
    """
    # Créer le répertoire si nécessaire
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le profil
    profiler.dump_stats(str(output_path))

    print(f"💾 Profil sauvegardé : {output_path}")


def print_top_functions(profiler: cProfile.Profile, n: int = 20) -> str:
    """
    Affiche et retourne les n fonctions les plus lentes.

    Args:
        profiler: Objet Profile de cProfile.
        n: Nombre de fonctions à afficher.

    Returns:
        String contenant le rapport.
    """
    # Créer un buffer pour capturer la sortie
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)

    # Trier par temps cumulatif et afficher le top n
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(n)

    output = s.getvalue()
    s.close()

    return output


def analyze_profiling_results(single_profiler: cProfile.Profile, batch_profiler: cProfile.Profile) -> str:
    """
    Analyse les résultats du profiling et génère un rapport.

    Args:
        single_profiler: Profile des prédictions individuelles.
        batch_profiler: Profile des prédictions batch.

    Returns:
        Rapport d'analyse (string).
    """
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("📊 RAPPORT DE PROFILING - BASELINE")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Section 1 : Prédictions individuelles
    report_lines.append("🔍 TOP 20 FONCTIONS - PRÉDICTIONS INDIVIDUELLES (1000x)")
    report_lines.append("-" * 80)
    single_stats = print_top_functions(single_profiler, n=20)
    report_lines.append(single_stats)
    report_lines.append("")

    # Section 2 : Prédictions batch
    report_lines.append("🔍 TOP 20 FONCTIONS - PRÉDICTIONS BATCH (10 batches × 100)")
    report_lines.append("-" * 80)
    batch_stats = print_top_functions(batch_profiler, n=20)
    report_lines.append(batch_stats)
    report_lines.append("")

    # Section 3 : Analyse des goulots d'étranglement
    report_lines.append("=" * 80)
    report_lines.append("🔴 GOULOTS D'ÉTRANGLEMENT IDENTIFIÉS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("D'après les métriques PostgreSQL (baseline_production.json) :")
    report_lines.append("  - Preprocessing : 27.96 ms (91.2% du temps total)")
    report_lines.append("  - Inference     : 2.71 ms (8.8% du temps total)")
    report_lines.append("")
    report_lines.append("Fonctions cibles pour optimisation (dans predictor.py) :")
    report_lines.append("  1. Label Encoding (lignes 127-134) - Boucle séquentielle sur 37 colonnes")
    report_lines.append("  2. One-Hot Encoding (lignes 137-149) - Boucles + pd.concat() répétés")
    report_lines.append("  3. Réordonnancement colonnes (ligne 152) - Regex sur 911 colonnes")
    report_lines.append("")
    report_lines.append("Recommandations :")
    report_lines.append("  → A1 : Vectoriser le label encoding avec df.replace()")
    report_lines.append("  → A2 : Utiliser ColumnTransformer pour one-hot encoding groupé")
    report_lines.append("  → A3 : Pré-calculer l'ordre des colonnes au __init__()")
    report_lines.append("")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)


def save_text_report(report: str, output_path: Path) -> None:
    """
    Sauvegarde le rapport texte.

    Args:
        report: Contenu du rapport.
        output_path: Chemin du fichier de sortie.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"📄 Rapport texte sauvegardé : {output_path}")


def main():
    """Fonction principale."""
    print("\n🚀 PROFILING API - PHASE 1.2\n")

    try:
        # Initialiser le prédicteur
        print("🔧 Initialisation du prédicteur...")
        predictor = CreditScoringPredictor()
        print("✅ Prédicteur chargé")

        # Profiler les prédictions individuelles
        single_profiler = profile_single_predictions(predictor, n=1000)
        save_profile(single_profiler, Path("reports/profiling/baseline_single.prof"))

        # Profiler les prédictions batch
        batch_profiler = profile_batch_predictions(predictor, n_batches=10, batch_size=100)
        save_profile(batch_profiler, Path("reports/profiling/baseline_batch.prof"))

        # Analyser les résultats
        print("\n📊 Analyse des résultats...")
        report = analyze_profiling_results(single_profiler, batch_profiler)

        # Afficher le rapport
        print("\n" + report)

        # Sauvegarder le rapport texte
        save_text_report(report, Path("reports/profiling/profiling_report.txt"))

        print("\n✅ Tâche 1.2 terminée : Profiling avec cProfile réalisé")
        print("📁 Fichiers générés :")
        print("   - reports/profiling/baseline_single.prof")
        print("   - reports/profiling/baseline_batch.prof")
        print("   - reports/profiling/profiling_report.txt")

        return 0

    except Exception as e:
        print(f"\n❌ Erreur : {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
