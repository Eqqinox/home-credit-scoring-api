"""
Script de benchmarking pour mesurer les performances du modèle optimisé.

Ce script mesure les performances de la version optimisée du predictor et
compare avec la baseline de production. Il génère des rapports JSON et des
graphiques de comparaison.

Usage:
    python src/scripts/benchmark.py

Output:
    - reports/benchmarks/baseline.json (copie de baseline_production.json)
    - reports/benchmarks/optimized.json (métriques version optimisée)
    - reports/benchmarks/comparison.json (comparaison baseline vs optimized)
    - reports/benchmarks/performance_comparison.png (bar chart)
    - reports/benchmarks/performance_boxplot.png (boxplot)
"""

import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Configurer matplotlib pour utiliser un backend non-interactif
matplotlib.use('Agg')

# Ajouter le répertoire src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.predictor import CreditScoringPredictor


def load_sample_data() -> Dict[str, Any]:
    """
    Charge un exemple de données pour le benchmarking.

    Returns:
        Dictionnaire représentant les features d'un client.
    """
    example_path = Path("example_single_request.json")

    if example_path.exists():
        with open(example_path, 'r') as f:
            data = json.load(f)
            return data
    else:
        print("⚠️  example_single_request.json non trouvé")
        raise FileNotFoundError("example_single_request.json requis pour le benchmarking")


def generate_batch_data(n: int = 100) -> List[Dict[str, Any]]:
    """
    Génère un batch de données pour le benchmarking.

    Args:
        n: Nombre de prédictions dans le batch.

    Returns:
        Liste de dictionnaires de features.
    """
    base_data = load_sample_data()
    batch = []

    for i in range(n):
        data_copy = base_data.copy()
        # Varier légèrement les valeurs numériques
        if "AMT_CREDIT" in data_copy:
            data_copy["AMT_CREDIT"] *= random.uniform(0.8, 1.2)
        if "AMT_INCOME_TOTAL" in data_copy:
            data_copy["AMT_INCOME_TOTAL"] *= random.uniform(0.8, 1.2)

        batch.append(data_copy)

    return batch


def benchmark_single_predictions(predictor: CreditScoringPredictor, n: int = 1000) -> Dict[str, List[float]]:
    """
    Mesure les performances sur n prédictions individuelles.

    Args:
        predictor: Instance du prédicteur.
        n: Nombre de prédictions à exécuter.

    Returns:
        Dictionnaire avec les timings (total_ms, preprocessing_ms, inference_ms).
    """
    print(f"\n⏱️  Benchmarking de {n} prédictions individuelles...")

    sample_data = load_sample_data()
    timings = {
        "total_ms": [],
        "preprocessing_ms": [],
        "inference_ms": []
    }

    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"   Progression: {i + 1}/{n} prédictions")

        start = time.perf_counter()
        result = predictor.predict(sample_data)
        end = time.perf_counter()

        total_time = (end - start) * 1000  # Convertir en ms
        timings["total_ms"].append(total_time)

        # Extraire les timings du result si disponibles
        if "preprocessing_time_ms" in result:
            timings["preprocessing_ms"].append(result["preprocessing_time_ms"])
        if "inference_time_ms" in result:
            timings["inference_ms"].append(result["inference_time_ms"])

    print(f"✅ Benchmark single predictions terminé ({n} prédictions)")
    return timings


def benchmark_batch_predictions(predictor: CreditScoringPredictor, n_batches: int = 10, batch_size: int = 100) -> Dict[str, List[float]]:
    """
    Mesure les performances sur n batches de prédictions.

    Args:
        predictor: Instance du prédicteur.
        n_batches: Nombre de batches à exécuter.
        batch_size: Taille de chaque batch.

    Returns:
        Dictionnaire avec les timings (total_ms, preprocessing_ms, inference_ms).
    """
    print(f"\n⏱️  Benchmarking de {n_batches} batches de {batch_size} prédictions...")

    timings = {
        "total_ms": [],
        "preprocessing_ms": [],
        "inference_ms": []
    }

    for i in range(n_batches):
        batch_data = generate_batch_data(batch_size)

        print(f"   Batch {i + 1}/{n_batches}")

        # Mesurer chaque prédiction du batch
        for data in batch_data:
            start = time.perf_counter()
            result = predictor.predict(data)
            end = time.perf_counter()

            total_time = (end - start) * 1000
            timings["total_ms"].append(total_time)

            if "preprocessing_time_ms" in result:
                timings["preprocessing_ms"].append(result["preprocessing_time_ms"])
            if "inference_time_ms" in result:
                timings["inference_ms"].append(result["inference_time_ms"])

    total_predictions = n_batches * batch_size
    print(f"✅ Benchmark batch predictions terminé ({total_predictions} prédictions)")
    return timings


def calculate_statistics(timings: List[float]) -> Dict[str, float]:
    """
    Calcule les statistiques complètes sur une liste de timings.

    Args:
        timings: Liste de temps en millisecondes.

    Returns:
        Dictionnaire avec mean, median, p95, p99, min, max, std.
    """
    arr = np.array(timings)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr))
    }


def calculate_throughput(timings: List[float]) -> Dict[str, float]:
    """
    Calcule le throughput (prédictions/seconde et /minute).

    Args:
        timings: Liste de temps en millisecondes.

    Returns:
        Dictionnaire avec predictions_per_second et predictions_per_minute.
    """
    mean_time_ms = np.mean(timings)
    mean_time_s = mean_time_ms / 1000

    predictions_per_second = 1 / mean_time_s if mean_time_s > 0 else 0
    predictions_per_minute = predictions_per_second * 60

    return {
        "predictions_per_second": float(predictions_per_second),
        "predictions_per_minute": float(predictions_per_minute)
    }


def generate_optimized_report(timings: Dict[str, List[float]], n_samples: int) -> Dict[str, Any]:
    """
    Génère le rapport de performance pour la version optimisée.

    Args:
        timings: Dictionnaire avec les timings mesurés.
        n_samples: Nombre total de prédictions.

    Returns:
        Dictionnaire formaté pour optimized.json.
    """
    report = {
        "source": "optimized_model",
        "n_samples": n_samples,
        "total_time_ms": calculate_statistics(timings["total_ms"]),
        "preprocessing_time_ms": calculate_statistics(timings["preprocessing_ms"]) if timings["preprocessing_ms"] else {},
        "inference_time_ms": calculate_statistics(timings["inference_ms"]) if timings["inference_ms"] else {},
        "throughput": calculate_throughput(timings["total_ms"])
    }

    # Calculer pourcentage preprocessing vs inference
    if timings["preprocessing_ms"] and timings["inference_ms"]:
        mean_prep = np.mean(timings["preprocessing_ms"])
        mean_inf = np.mean(timings["inference_ms"])
        mean_total = np.mean(timings["total_ms"])

        report["preprocessing_time_ms"]["percentage_of_total"] = (mean_prep / mean_total) * 100
        report["inference_time_ms"]["percentage_of_total"] = (mean_inf / mean_total) * 100

    return report


def load_baseline_report(baseline_path: Path) -> Dict[str, Any]:
    """
    Charge le rapport baseline depuis baseline_production.json.

    Args:
        baseline_path: Chemin vers baseline_production.json.

    Returns:
        Dictionnaire du rapport baseline.
    """
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline non trouvée: {baseline_path}")

    with open(baseline_path, 'r') as f:
        return json.load(f)


def generate_comparison_report(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère le rapport de comparaison baseline vs optimized.

    Args:
        baseline: Rapport baseline.
        optimized: Rapport optimized.

    Returns:
        Dictionnaire avec les améliorations calculées.
    """
    comparison = {
        "baseline_source": baseline.get("source", "unknown"),
        "optimized_source": optimized.get("source", "unknown"),
        "baseline_samples": baseline.get("n_samples", 0),
        "optimized_samples": optimized.get("n_samples", 0),
        "improvements": {}
    }

    # Calculer les améliorations pour chaque métrique
    metrics = ["mean", "median", "p95", "p99", "min", "max"]

    for metric in metrics:
        # Total time
        baseline_total = baseline.get("total_time_ms", {}).get(metric, 0)
        optimized_total = optimized.get("total_time_ms", {}).get(metric, 0)

        if baseline_total > 0:
            reduction_percent = ((baseline_total - optimized_total) / baseline_total) * 100
            comparison["improvements"][f"total_{metric}_reduction_percent"] = round(reduction_percent, 2)
            comparison["improvements"][f"total_{metric}_baseline_ms"] = round(baseline_total, 2)
            comparison["improvements"][f"total_{metric}_optimized_ms"] = round(optimized_total, 2)

        # Preprocessing time
        baseline_prep = baseline.get("preprocessing_time_ms", {}).get(metric, 0)
        optimized_prep = optimized.get("preprocessing_time_ms", {}).get(metric, 0)

        if baseline_prep > 0:
            reduction_percent = ((baseline_prep - optimized_prep) / baseline_prep) * 100
            comparison["improvements"][f"preprocessing_{metric}_reduction_percent"] = round(reduction_percent, 2)
            comparison["improvements"][f"preprocessing_{metric}_baseline_ms"] = round(baseline_prep, 2)
            comparison["improvements"][f"preprocessing_{metric}_optimized_ms"] = round(optimized_prep, 2)

    # Throughput improvement
    baseline_throughput = baseline.get("throughput", {}).get("predictions_per_second", 0)
    optimized_throughput = optimized.get("throughput", {}).get("predictions_per_second", 0)

    if baseline_throughput > 0:
        throughput_increase = ((optimized_throughput - baseline_throughput) / baseline_throughput) * 100
        comparison["improvements"]["throughput_increase_percent"] = round(throughput_increase, 2)
        comparison["improvements"]["throughput_baseline_pred_per_sec"] = round(baseline_throughput, 2)
        comparison["improvements"]["throughput_optimized_pred_per_sec"] = round(optimized_throughput, 2)

    return comparison


def create_bar_chart(baseline: Dict[str, Any], optimized: Dict[str, Any], output_path: Path):
    """
    Crée un bar chart comparant baseline vs optimized.

    Args:
        baseline: Rapport baseline.
        optimized: Rapport optimized.
        output_path: Chemin de sortie pour le PNG.
    """
    print("\n📊 Génération du bar chart...")

    metrics = ["mean", "median", "p95", "p99"]
    baseline_values = [baseline["total_time_ms"][m] for m in metrics]
    optimized_values = [optimized["total_time_ms"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#e74c3c')
    bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimized', color='#27ae60')

    ax.set_xlabel('Métrique', fontsize=12)
    ax.set_ylabel('Temps (ms)', fontsize=12)
    ax.set_title('Comparaison Performance : Baseline vs Optimized', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Ajouter les valeurs au-dessus des barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Bar chart sauvegardé: {output_path}")


def create_boxplot(baseline: Dict[str, Any], optimized: Dict[str, Any],
                   baseline_timings: List[float], optimized_timings: List[float],
                   output_path: Path):
    """
    Crée un boxplot comparant les distributions baseline vs optimized.

    Args:
        baseline: Rapport baseline (pour le titre).
        optimized: Rapport optimized (pour le titre).
        baseline_timings: Timings bruts baseline (échantillon).
        optimized_timings: Timings bruts optimized.
        output_path: Chemin de sortie pour le PNG.
    """
    print("\n📊 Génération du boxplot...")

    # Utiliser les statistiques du baseline (on n'a pas les timings bruts de la prod)
    # On va simuler une distribution à partir des statistiques
    baseline_stats = baseline["total_time_ms"]
    baseline_sample = np.random.normal(
        loc=baseline_stats["mean"],
        scale=baseline_stats["std"],
        size=1000
    )
    # Clipper pour respecter min/max
    baseline_sample = np.clip(baseline_sample, baseline_stats["min"], baseline_stats["max"])

    fig, ax = plt.subplots(figsize=(10, 6))

    data = [baseline_sample, optimized_timings]
    labels = ['Baseline\n(Production)', 'Optimized\n(Version actuelle)']

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    # Coloriser les boîtes
    colors = ['#e74c3c', '#27ae60']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Temps Total (ms)', fontsize=12)
    ax.set_title('Distribution des Temps de Réponse : Baseline vs Optimized',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Ajouter statistiques texte
    baseline_mean = baseline_stats["mean"]
    optimized_mean = np.mean(optimized_timings)
    reduction = ((baseline_mean - optimized_mean) / baseline_mean) * 100

    textstr = f'Baseline Mean: {baseline_mean:.2f} ms\nOptimized Mean: {optimized_mean:.2f} ms\nRéduction: {reduction:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Boxplot sauvegardé: {output_path}")


def main():
    """
    Point d'entrée principal du script de benchmarking.
    """
    print("=" * 70)
    print("🚀 BENCHMARKING - VERSION OPTIMISÉE")
    print("=" * 70)

    # 1. Initialiser le predictor
    print("\n📦 Chargement du predictor optimisé...")
    predictor = CreditScoringPredictor()
    print("✅ Predictor chargé avec succès")

    # 2. Benchmark single predictions (1000)
    timings_single = benchmark_single_predictions(predictor, n=1000)

    # 3. Benchmark batch predictions (10 batches × 100)
    timings_batch = benchmark_batch_predictions(predictor, n_batches=10, batch_size=100)

    # 4. Combiner les timings (total 2000 prédictions)
    combined_timings = {
        "total_ms": timings_single["total_ms"] + timings_batch["total_ms"],
        "preprocessing_ms": timings_single["preprocessing_ms"] + timings_batch["preprocessing_ms"],
        "inference_ms": timings_single["inference_ms"] + timings_batch["inference_ms"]
    }

    total_samples = len(combined_timings["total_ms"])
    print(f"\n📊 Total prédictions mesurées: {total_samples}")

    # 5. Générer le rapport optimized
    optimized_report = generate_optimized_report(combined_timings, total_samples)

    # 6. Charger le rapport baseline
    baseline_path = Path("reports/benchmarks/baseline_production.json")
    print(f"\n📂 Chargement du baseline depuis: {baseline_path}")
    baseline_report = load_baseline_report(baseline_path)

    # 7. Générer le rapport de comparaison
    comparison_report = generate_comparison_report(baseline_report, optimized_report)

    # 8. Créer le répertoire de sortie
    output_dir = Path("reports/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 9. Sauvegarder les rapports JSON
    print("\n💾 Sauvegarde des rapports JSON...")

    # Copier baseline_production.json vers baseline.json
    baseline_output = output_dir / "baseline.json"
    with open(baseline_output, 'w') as f:
        json.dump(baseline_report, f, indent=2)
    print(f"   ✅ {baseline_output}")

    # Sauvegarder optimized.json
    optimized_output = output_dir / "optimized.json"
    with open(optimized_output, 'w') as f:
        json.dump(optimized_report, f, indent=2)
    print(f"   ✅ {optimized_output}")

    # Sauvegarder comparison.json
    comparison_output = output_dir / "comparison.json"
    with open(comparison_output, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    print(f"   ✅ {comparison_output}")

    # 10. Générer les graphiques
    bar_chart_path = output_dir / "performance_comparison.png"
    create_bar_chart(baseline_report, optimized_report, bar_chart_path)

    boxplot_path = output_dir / "performance_boxplot.png"
    create_boxplot(baseline_report, optimized_report,
                   [], combined_timings["total_ms"],  # Pas de timings bruts baseline
                   boxplot_path)

    # 11. Afficher résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES RÉSULTATS")
    print("=" * 70)

    print("\n🎯 BASELINE (Production - 1,166 prédictions)")
    print(f"   Total Mean    : {baseline_report['total_time_ms']['mean']:.2f} ms")
    print(f"   Total Median  : {baseline_report['total_time_ms']['median']:.2f} ms")
    print(f"   Total P95     : {baseline_report['total_time_ms']['p95']:.2f} ms")
    print(f"   Throughput    : {baseline_report['throughput']['predictions_per_second']:.2f} pred/sec")

    print("\n🚀 OPTIMIZED (Version actuelle - 2,000 prédictions)")
    print(f"   Total Mean    : {optimized_report['total_time_ms']['mean']:.2f} ms")
    print(f"   Total Median  : {optimized_report['total_time_ms']['median']:.2f} ms")
    print(f"   Total P95     : {optimized_report['total_time_ms']['p95']:.2f} ms")
    print(f"   Throughput    : {optimized_report['throughput']['predictions_per_second']:.2f} pred/sec")

    print("\n📈 AMÉLIORATIONS")
    print(f"   Réduction Mean     : {comparison_report['improvements']['total_mean_reduction_percent']:.2f}%")
    print(f"   Réduction Median   : {comparison_report['improvements']['total_median_reduction_percent']:.2f}%")
    print(f"   Réduction P95      : {comparison_report['improvements']['total_p95_reduction_percent']:.2f}%")
    print(f"   Augmentation Throughput : {comparison_report['improvements']['throughput_increase_percent']:.2f}%")

    print("\n" + "=" * 70)
    print("✅ BENCHMARKING TERMINÉ AVEC SUCCÈS")
    print("=" * 70)

    # Vérifier l'objectif -40% minimum
    mean_reduction = comparison_report['improvements']['total_mean_reduction_percent']
    if mean_reduction >= 40:
        print(f"\n🎉 OBJECTIF ATTEINT : Réduction de {mean_reduction:.1f}% (objectif: -40% minimum)")
    else:
        print(f"\n⚠️  OBJECTIF NON ATTEINT : Réduction de {mean_reduction:.1f}% (objectif: -40% minimum)")


if __name__ == "__main__":
    main()
