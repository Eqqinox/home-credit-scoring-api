# Home Credit - API de Scoring Crédit (MLOps) 

> Déploiement d'un modèle de scoring crédit en production avec approche MLOps.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.122.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52.1-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-python:3.11--slim-2496ED.svg)](https://hub.docker.com/_/python)
[![MLflow](https://img.shields.io/badge/MLflow-3.6.0-orange.svg)](https://mlflow.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)](https://lightgbm.readthedocs.io/)
---

## Table des matières

1. [Contexte du projet](#contexte-du-projet)
2. [Architecture](#architecture)
3. [Optimisations](#optimisations)
4. [Structure du projet](#structure-du-projet)
5. [Installation](#installation)
6. [Utilisation](#utilisation)
7. [API](#api)
8. [Dashboard Monitoring](#dashboard-monitoring)
9. [Tests](#tests)
10. [Technologies](#technologies)
11. [Auteur](#auteur)

---

## Contexte du projet

**"Prêt à dépenser"** est une société financière proposant des crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt.

### Mission

Développer un **outil de scoring crédit** pour :
- Calculer la probabilité qu'un client rembourse son crédit
- Classifier automatiquement les demandes (accepter/refuser)
- Monitorer les performances du modèle en production
- Détecter les dérives de données (data drift)

### Contrainte métier

Le coût d'un **Faux Négatif** (mauvais client accepté) est **10x** supérieur au coût d'un **Faux Positif** (Optimisation du seuil de décision pour minimiser le coût métier total).

---

## Architecture

```
                         ┌──────────────────┐     
                         │ Structlog (JSON) │
                         │(Logs temps réel) │
                         └──────────────────┘
                                  ^
                                  │
                                  │ 
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐      ┌───────────────────────────────┐
│   Client HTTP   │─────>│   FastAPI API    │─────>│   PostgreSQL    │─────>│         Evidently AI          │
│                 │<─────│   (Port 8000)    │<─────│   (Stockage)    │<─────│   Rapports Drift (HTML/JSON)  │
└─────────────────┘      └──────────────────┘      └─────────────────┘      └───────────────────────────────┘
                                  │                         │               
                                  │                         │
                                  v                         v
                         ┌──────────────────┐      ┌─────────────────┐
                         │  LightGBM Model  │      │    Streamlit    │
                         │    (Scoring)     │      │    Dashboard    │
                         └──────────────────┘      │   (Port 8501)   │
                                                   └─────────────────┘

```

### Composants

1. **API FastAPI** : 5 endpoints REST (predict, batch, health, model-info, docs)
2. **Modèle LightGBM** : Scoring crédit optimisé (AUC = 0.7828, seuil = 0.5225)
3. **PostgreSQL** : Stockage prédictions + features + drift reports (4 tables)
4. **Streamlit** : Dashboard monitoring temps réel (5 pages)
5. **Structlog** : Logging structuré JSON (temps réel)
6. **Evidently AI** : Détection data drift automatique
7. **GitHub Actions** : Pipeline CI/CD automatisé (4 jobs : test, build, push, deploy)
8. **Docker** : Conteneurisation (python:3.11-slim)
9. **Tests** : 155 tests automatisés (89% de couverture)
---

 ## Optimisations

  **Objectif** : Réduire la latence de l'API

  **Méthodologie** :
  1. Profiling avec `cProfile` (sur 2,000 prédictions)
  2. Identification de 3 goulots d'étranglement (preprocessing 91.2% du temps)
  3. Implémentation de 3 optimisations ciblées
  4. Benchmarking quantitatif avec graphiques

  ### Comparaison Baseline vs Optimized

  | Métrique | Baseline (Production) | Optimized | Amélioration |
  |----------|----------------------|-----------|--------------|
  | **Mean** | 30.67 ms | 17.55 ms | **-42.78%** |
  | **Median (P50)** | 30.49 ms | 17.27 ms | **-43.35%** |
  | **P95** | 32.45 ms | 17.83 ms | **-45.06%** |
  | **P99** | 35.11 ms | 18.33 ms | **-47.79%** |
  | **Throughput** | 32.61 pred/sec | 56.98 pred/sec | **+74.73%** |

  **Source** :
  - Baseline : 1,166 prédictions production (PostgreSQL 09/12 → 16/12/2025)
  - Optimized : 2,000 prédictions benchmarking (16/12/2025)

  ### Optimisations Implémentées

  | ID | Optimisation | Description |
  |----|--------------|-------------|
  | **A1** | Label Encoding Vectorisé | Pré-calcul mappings + `df.replace()` pandas au lieu de `LabelEncoder.transform()` sklearn |
  | **A2** | One-Hot Encoding Groupé | UN SEUL `pd.concat()` au lieu de 32 (réduction O(n²) → O(n)) |
  | **A3** | Caching Colonnes Finales | Pré-calcul ordre colonnes finales (élimination regex sur 911 cols) |

  **Gain cumulé mesuré** : **-42.78%**

  ### Impact Business

  - **UX améliorée** : Réponse quasi-instantanée (< 20 ms pour 99% des clients)
  - **Scalabilité** : +75% de capacité sans upgrade matériel (4.9M pred/jour vs 2.8M)
  - **Coûts réduits** : -43% temps CPU par prédiction

  ### Documentation

  Rapport complet d'optimisation : [`docs/OPTIMIZATION_REPORT.md`](docs/OPTIMIZATION_REPORT.md)

  **Graphiques (non versionnés)** :
  - `reports/benchmarks/performance_comparison.png` (bar chart)
  - `reports/benchmarks/performance_boxplot.png` (distributions)
---

## Structure du projet

```
home-credit-scoring-api/
  ├── .github/
  │   └── workflows/
  │       └── ci-cd.yml                     # Pipeline CI/CD
  ├── data/                                 
  │   └── README.md                         # Documentation dataset (non versionné)    
  ├── docs/
  │   ├── OPTIMIZATION_REPORT.md            # Rapport optimisations
  │   └── RGPD_COMPLIANCE.md                # Conformité RGPD
  ├── models/                              
  │   ├── model.pkl                         # LightGBM optimisé
  │   ├── feature_names.pkl                 # Noms features (911)
  │   ├── label_encoders.pkl                # Encodeurs catégoriels
  │   ├── onehot_encoder.pkl                # Encodeur one-hot
  │   ├── metrics.json                      # Métriques modèle (AUC, etc.)
  │   └── threshold.json                    # Seuil optimal (0.5225)
  ├── notebooks/
  │   ├── 01_Modelisation_MLflow.ipynb      # Expérimentations MLflow (Projet partie 1)
  │   ├── 02_model_production.ipynb         # Préparation modèle production
  │   ├── 03_test_model_loading.ipynb       # Tests chargement modèle
  │   └── 04_drift_analysis.ipynb           # Analyse data drift (Evidently)
  ├── src/
  │   ├── api/
  │   │   ├── __init__.py
  │   │   ├── config.py                     # Configuration (Pydantic Settings)
  │   │   ├── main.py                       # FastAPI application (5 endpoints)
  │   │   ├── predictor.py                  # Logique ML + optimisations
  │   │   ├── preprocessing.py              # Utilitaires (legacy)
  │   │   └── schemas.py                    # Pydantic models (validation)
  │   ├── monitoring/
  │   │   ├── __init__.py
  │   │   ├── dashboard.py                  # Page d'accueil Streamlit
  │   │   ├── drift_detector.py             # Détection drift (Evidently) 
  │   │   ├── logger.py                     # Logging structuré (structlog)
  │   │   ├── storage.py                    # PostgreSQL ORM (SQLAlchemy)
  │   │   └── pages/
  │   │       ├── business.py               # Profils clients + montants
  │   │       ├── drift.py                  # Rapports drift (HTML)
  │   │       ├── overview.py               # KPIs + filtres temporels
  │   │       └── performance.py            # Latences + erreurs
  │   └── scripts/
  │       ├── __init__.py
  │       ├── benchmark.py                  # Benchmarking performances
  │       ├── calculate_baseline_metrics.py # Extraction métriques production
  │       ├── generate_drift_report.py      # Génération rapports drift
  │       ├── init_database.py              # Initialisation PostgreSQL
  │       ├── profile_api.py                # Profiling cProfile 
  │       └── simulate_traffic.py           # Simulation trafic API
  ├── tests/
  │   ├── __init__.py
  │   ├── conftest.py                       # Fixtures pytest
  │   ├── test_additional_coverage.py       # Tests edge cases + gestion erreurs
  │   ├── test_api_endpoints.py             # Tests endpoints FastAPI
  │   ├── test_predictor.py                 # Tests modèle ML
  │   ├── test_validation.py                # Tests validation données
  │   └── monitoring/
  │       ├── __init__.py
  │       ├── test_drift_detector.py        # Tests détection drift
  │       ├── test_logger.py                # Tests logging structuré
  │       └── test_storage.py               # Tests PostgreSQL ORM
  ├── .coveragerc-ci                        # Config coverage CI/CD
  ├── .dockerignore                         # Exclusions Docker
  ├── .gitignore                            # Exclusions Git
  ├── API_USAGE.md                          # Guide utilisation API
  ├── Dockerfile                            # Image Docker production
  ├── Dockerfile.huggingface                # Image Docker Hugging Face
  ├── MONITORING.md                         # Guide monitoring
  ├── README.md                             # Documentation principale
  ├── docker-compose.yml                    # Orchestration locale
  ├── example_batch_request.json            # Exemple API (3 clients)
  ├── example_single_request.json           # Exemple API (1 client)
  ├── pyproject.toml                        # Dépendances + config outils
  ├── requirements.txt                      # Dépendances production (compilées)
  └── uv.lock                               # Lockfile uv
```
---

## Installation

### Prérequis

- **Python 3.11+**
- **PostgreSQL 16** (pour stockage production)
- **UV package manager** ou pip
- **Git**
- **Docker**

### Installation avec UV

```bash
# Cloner le repository
git clone https://github.com/Eqqinox/home-credit-scoring-api.git
cd home-credit-scoring-api

# Créer et activer l'environnement virtuel
uv venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Installer les dépendances
uv pip install -r requirements.txt
```

### Configuration PostgreSQL

```bash
# Lancer PostgreSQL (macOS avec Homebrew)
brew services start postgresql@16

# Initialiser la base de données
python src/scripts/init_database.py
```

**4 tables créées** :
- `predictions` (15 colonnes) : Prédictions individuelles
- `feature_values` : Top 20 features par prédiction
- `anomalies` : Logs d'erreurs API
- `drift_reports` : Rapports Evidently AI

---

## Utilisation

### 1. Lancer l'API FastAPI

```bash
# Mode local
ENVIRONMENT=local LOG_LEVEL=INFO uvicorn src.api.main:app --reload --port 8000

# Mode production (JSON structuré)
ENVIRONMENT=production LOG_LEVEL=INFO uvicorn src.api.main:app --port 8000
```

**Accès** :
- API : http://localhost:8000
- Swagger UI : http://localhost:8000/docs
- Redoc : http://localhost:8000/redoc

### 2. Lancer le Dashboard Streamlit

```bash
streamlit run src/monitoring/dashboard.py --server.port 8501
```

**Accès** : http://localhost:8501

### 3. Générer du Trafic (Simulation)

```bash
# Simulation de 100 prédictions avec drift
python src/scripts/simulate_traffic.py --num-predictions 100 --delay 0.5 --drift-prob 0.3
```

**Options** :
- `--num-predictions` : Nombre de prédictions (défaut : 10)
- `--delay` : Délai entre requêtes en secondes (défaut : 0.5)
- `--drift-prob` : Probabilité d'appliquer du drift (défaut : 0.3)
- `--drift-magnitude` : Magnitude du drift ±% (défaut : 0.15)

---

## API

### Endpoints disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Health check de l'API |
| GET | `/model-info` | Informations sur le modèle |
| POST | `/predict` | Prédiction pour un client |
| POST | `/predict-batch` | Prédictions en batch (max 100) |
| GET | `/docs` | Documentation Swagger UI |

### Exemple de requête

**Prédiction simple** :

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @example_single_request.json
```

**Réponse** :

```json
{
  "client_id": 100001,
  "probability_default": 0.3521,
  "prediction": 0,
  "decision": "approve",
  "risk_level": "MEDIUM",
  "threshold_used": 0.5225,
  "model_version": "1.0.0"
}
```

**Voir `API_USAGE.md` pour plus d'exemples.**

---

## Dashboard Monitoring

Le dashboard Streamlit offre **5 pages** de monitoring :

### Page - dashboard
- Statut API FastAPI
- Statut PostgreSQL
- Métriques globales
- Guide d'utilisation

### Page - business
- Pie chart : Profils clients (Approve/Refuse)
- Histogram : Distribution des montants de crédit

### Page - drift
- **4 KPIs** : Drift détecté (OUI/NON), score de drift, features affectées, seuil alerte
- **Rapport HTML interactif** : Visualisations Evidently AI (distributions, tests statistiques)
- **Historique** : Line chart évolution des scores de drift dans le temps
- **Génération** : Commande `python src/scripts/generate_drift_report.py --days 7`

**Seuil d'alerte** : 30% des features avec drift → Réentraînement recommandé

**Note** : Le seuil 0.5 affiché dans les rapports HTML correspond au p-value des tests statistiques d'Evidently AI (non configurable). Le seuil d'alerte de 30% s'applique au niveau global.

### Page - overview
- **4 KPIs** : Total prédictions, taux approbation, latence moyenne, taux erreur
- **Filtres temporels** : 24h, 7j, 30j, Tout
- **5 visualisations** :
  - Donut chart : Répartition Approve/Refuse
  - Line chart : Volume de prédictions par heure
  - Histogram : Distribution des probabilités (avec seuil 0.5225)
  - Bar chart : Niveaux de confiance (LOW/MEDIUM/HIGH)

### Page - performance
- Boxplot : Distribution des latences par endpoint
- Top 10 : Requêtes les plus lentes
- Tableau : Erreurs HTTP (code != 200)


---

## Tests

### Lancer tous les tests

```bash
pytest tests/ -v --cov=src
```

**Couverture actuelle : 89%**

### Tests par module

```bash
# Tests API
pytest tests/test_api_endpoints.py -v

# Tests logger
pytest tests/monitoring/test_logger.py -v

# Tests storage
pytest tests/monitoring/test_storage.py -v
```

### Tests en conditions réelles

```bash
# Vérifier stockage PostgreSQL
psql -U moon -d credit_scoring_prod -c "SELECT COUNT(*) FROM predictions;"

# Voir les statistiques
python3 -c "
from src.monitoring.storage import PredictionStorage
storage = PredictionStorage(database_url='postgresql://moon:moon@localhost:5432/credit_scoring_prod')
import json
print(json.dumps(storage.get_stats(), indent=2))
storage.close()
"
```

---

## Docker

### Build et run local

```bash
# Build l'image
docker build -t credit-scoring-api .

# Run le container
docker run -p 8000:8000 credit-scoring-api
```

### Docker Compose (avec PostgreSQL)

```bash
# Lancer tous les services
docker-compose up -d

# Voir les logs
docker-compose logs -f api

# Arrêter
docker-compose down
```

---

## Technologies

### Data Science & ML
- **Pandas**, **NumPy** : Manipulation de données
- **Scikit-learn** : Preprocessing, métriques
- **LightGBM** : Modèle de scoring
- **MLflow** : Tracking expérimentations (partie 1 du projet)

### Backend & API
- **FastAPI** : API REST
- **Pydantic** : Validation des données
- **Uvicorn** : Serveur ASGI

### Database & Storage
- **PostgreSQL 16** : Base de données production
- **SQLAlchemy** : ORM + Connection pooling
- **Psycopg2** : Driver PostgreSQL

### Monitoring & Logging
- **Streamlit** : Dashboard interactif
- **Plotly** : Visualisations interactives
- **structlog** : Logging structuré JSON
- **Evidently AI** : Détection data drift

### Testing & CI/CD
- **Pytest** : Tests unitaires + intégration
- **pytest-cov** : Couverture de code
- **pytest-asyncio** : Tests async
- **GitHub Actions** : Pipeline CI/CD

### DevOps
- **Docker** : Conteneurisation
- **docker-compose** : Orchestration
- **UV** : Package manager
- **Hugging Face Spaces** : Déploiement cloud

---

## Résultats du modèle

### Modèle sélectionné : LightGBM

  | Métrique                      | Valeur  |
  |-------------------------------|---------|
  | **AUC ROC**                   | 0.7828  |
  | **Seuil optimal**             | 0.5225  |
  | **Recall (seuil optimal)**    | 0.6389  |
  | **Precision (seuil optimal)** | 0.2023  |
  | **Temps d'entraînement**      | 18.3s   |
  | **Features**                  | 911     |
  | **Échantillons train**        | 246,008 |
  | **Échantillons validation**   | 61,503  |

  **Contrainte métier** : Coût FN = 10x Coût FP

### Coût métier (optimisation)

  | Métrique         | Seuil par défaut (0.5) | Seuil optimal (0.5225) | Amélioration |
  |------------------|------------------------|------------------------|--------------|
  | **Coût total**   | 30,490                 | 30,437                 | **-53**      |
  | **Coût FN**      | 16,740                 | 17,930                 | +1,190       |
  | **Coût FP**      | 13,750                 | 12,507                 | **-1,243**   |
  | **Faux Négatifs**| 1,674                  | 1,793                  | +119         |
  | **Faux Positifs**| 13,750                 | 12,507                 | **-1,243**   |

---

## Commandes utiles

### PostgreSQL

```bash
# Se connecter
psql -U moon -d credit_scoring_prod

# Voir les tables
\dt

# Compter les prédictions
SELECT COUNT(*) FROM predictions;

# Stats par décision
SELECT decision, COUNT(*) FROM predictions GROUP BY decision;
```

### MLflow

```bash
mlflow ui --port 5000
```

### Tests

```bash
# Tous les tests avec couverture
pytest tests/ -v --cov=src --cov-report=html

# Rapport HTML
open htmlcov/index.html
```

---

## Documentation

- **API** : `API_USAGE.md` - Guide d'utilisation
- **Monitoring** : `MONITORING.md` - Guide système de monitoring et détection drift
- **Swagger UI** : http://localhost:8000/docs
- **Redoc** : http://localhost:8000/redoc

---

## Liens

- **Repository** : https://github.com/Eqqinox/home-credit-scoring-api
- **API Live** : https://eqqinox-credit-scoring-api.hf.space
- **Swagger Live** : https://eqqinox-credit-scoring-api.hf.space/docs
- **Kaggle** : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)

---

## Licence

MIT License

---

## Auteur

**Mounir Meknaci**

- Email : meknaci81@gmail.com
- LinkedIn : [Mounir Meknaci](https://www.linkedin.com/in/mounir-meknaci/)
- Formation : Expert en ingénierie et science des données
- Projet : Home Credit Default Risk - Approche MLOps

---

*Dernière mise à jour: Décembre 2025*  
*Projet Home Credit Scoring API - OpenClassrooms*.  
*Auteur : Mounir Meknaci*.  
*Version : 1.0*