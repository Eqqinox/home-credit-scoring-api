# Home Credit - API de Scoring Crédit (MLOps)

> Déploiement d'un modèle de scoring crédit en production avec approche MLOps complète

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)

---

## Contexte du projet

**"Prêt à dépenser"** est une société financière proposant des crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt.

L'entreprise souhaite mettre en œuvre un **outil de "scoring crédit"** pour :
- Calculer automatiquement la probabilité qu'un client rembourse son crédit
- Classifier les demandes en crédit accordé ou refusé
- Améliorer la transparence des décisions de crédit

---

## Objectifs

### Partie 1 - Développement du modèle (Terminée)
- ✅ Exploration et nettoyage des données (307k clients, 646 features)
- ✅ Feature engineering et agrégation des tables
- ✅ Entraînement et comparaison de modèles avec MLflow
- ✅ Sélection du meilleur modèle : **LightGBM** (AUC = 0.76)
- ✅ Optimisation des hyperparamètres et du seuil métier

### Partie 2 - Mise en production (En cours)
- **Étape 1** : Structure du projet et versioning Git
- **Étape 2** : API REST et pipeline CI/CD
- **Étape 3** : Stockage et analyse des données de production
- **Étape 4** : Monitoring des performances et data drift

---

## 📁 Structure du projet
```
Projet8/
├── data/                        # Données (non versionnées)
│   ├── README.md               # Documentation sur les données
│   └── app_train_models.csv    # Dataset préparé (non committé)
│
├── notebooks/                   # Analyses et expérimentations
│   └── 01_Modelisation_MLflow.ipynb  # Notebook de modélisation
│
├── src/                         # Code source
│   ├── models/                 # Entraînement et inférence
│   │   ├── train.py           # Script d'entraînement
│   │   └── predict.py         # Script de prédiction
│   └── utils/                  # Utilitaires
│       └── metrics.py          # Métriques métier
│
├── tests/                       # Tests unitaires
│
├── models/                      # Modèles sauvegardés
│   └── .gitkeep
│
├── pyproject.toml              # Configuration UV et dépendances
├── uv.lock                     # Lock file UV
├── requirements.txt            # Dépendances (compatibilité)
├── Dockerfile                  # Conteneurisation
└── README.md                   # Ce fichier
```

---

## Installation

### Prérequis
- Python 3.11
- [UV package manager](https://github.com/astral-sh/uv) (recommandé) ou pip
- Git

### Installation avec UV (recommandé)
```bash
# Cloner le repository
git clone https://github.com/[votre-username]/home-credit-scoring-api.git
cd home-credit-scoring-api

# Créer et activer l'environnement virtuel
uv venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Installer les dépendances
uv pip install -r requirements.txt
```

### Installation avec pip (alternative)
```bash
# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

---

## Données

Les données doivent être placées dans le dossier `data/`.

**Voir `data/README.md`** pour obtenir les données et comprendre leur structure.

**Important** : Le fichier `app_train_models.csv` n'est pas versionné dans Git (trop volumineux).

---

## Notebook de modélisation

### `notebooks/01_Modelisation_MLflow.ipynb`

Ce notebook documente la phase de modélisation complète :

1. **Configuration MLflow** avec backend SQLite
2. **Chargement des données** préparées (646 features)
3. **Baseline model** (DummyClassifier)
4. **Test de plusieurs modèles** :
   - Logistic Regression
   - Random Forest
   - XGBoost
   - LightGBM
5. **Comparaison et sélection** du meilleur modèle
6. **Optimisation des hyperparamètres**
7. **Optimisation du seuil métier** (coût FN = 10x coût FP)
8. **Enregistrement dans MLflow Model Registry**

Pour lancer le notebook :
```bash
jupyter notebook notebooks/01_Modelisation_MLflow.ipynb
```

---

## Résultats du modèle

### Comparaison des modèles testés

| Modèle | AUC ROC (Validation) | Business Score | Temps d'entraînement |
|--------|---------------------|----------------|---------------------|
| Dummy Classifier | 0.50 | - | 1s |
| Logistic Regression | 0.71 | 0.65 | 45s |
| Random Forest | 0.73 | 0.68 | 120s |
| XGBoost | 0.75 | 0.71 | 180s |
| **LightGBM ** | **0.76** | **0.73** | **90s** |

### Modèle sélectionné : LightGBM

**Critères de sélection :**
- ✅ Meilleur AUC ROC (0.76)
- ✅ Meilleur Business Score (0.73)
- ✅ Temps d'entraînement raisonnable (90s)
- ✅ Pas d'overfitting (cohérence CV vs Validation)

**Contrainte métier :**
- Coût d'un Faux Négatif (FN) = 10x le coût d'un Faux Positif (FP)
- Seuil de décision optimisé en conséquence

---

## Technologies

### Data Science & Machine Learning
- **Manipulation de données** : Pandas, NumPy
- **Visualisation** : Matplotlib, Seaborn
- **Machine Learning** : Scikit-learn, LightGBM, XGBoost
- **Optimisation** : Optuna
- **Interprétabilité** : SHAP

### MLOps
- **Tracking d'expériences** : MLflow
- **API** : FastAPI, Gradio
- **Monitoring** : Streamlit, Evidently
- **Tests** : Pytest
- **Conteneurisation** : Docker
- **Package Manager** : UV

---

## Tests (à venir - Étape 2)
```bash
pytest tests/
```

---

## Docker (à venir - Étape 2)
```bash
# Build
docker build -t credit-scoring-api .

# Run
docker run -p 8000:8000 credit-scoring-api
```

---

## Licence

MIT License

---

## Auteur

**Mounir Meknaci**

- 📧 Email : [meknaci81@gmail]
- 💼 LinkedIn : [https://www.linkedin.com/in/mounir-meknaci/]
- 🎓 Formation : Data Scientist / ML Engineer
- 📂 Projet : Home Credit Default Risk - Approche MLOps