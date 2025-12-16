# 📊 Guide de Monitoring - Credit Scoring API

> Documentation complète du système de monitoring et détection de drift

---

## 📋 Table des Matières

1. [Architecture](#architecture)
2. [Lancer le Dashboard](#lancer-le-dashboard)
3. [Interpréter les Métriques](#interpréter-les-métriques)
4. [Configuration des Alertes](#configuration-des-alertes)
5. [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture

### Vue d'ensemble du système de monitoring

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTÈME DE MONITORING                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │         FastAPI Application              │
        │         (Port 8000)                      │
        │                                          │
        │  • Endpoints : /predict, /predict-batch  │
        │  • Logging structuré (structlog)         │
        │  • Mesures de performance                │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │       PredictionStorage (ORM)            │
        │                                          │
        │  • Connection pooling (5 + 10 overflow)  │
        │  • Stockage prédictions + features       │
        │  • Calculs automatiques (confidence,     │
        │    data_quality_score)                   │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │         PostgreSQL 16                    │
        │      (credit_scoring_prod)               │
        │                                          │
        │  Tables :                                │
        │  • predictions (15 colonnes)             │
        │  • feature_values (top 20 features)      │
        │  • anomalies (logs erreurs)              │
        │  • drift_reports (Evidently AI)          │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │      Streamlit Dashboard                 │
        │         (Port 8501)                      │
        │                                          │
        │  Pages :                                 │
        │  🏠 Accueil (statut système)             │
        │  📈 Overview (KPIs + graphiques)         │
        │  ⚡ Performance (latences + erreurs)     │
        │  💼 Business (profils + montants)        │
        │  🔍 Data Drift (rapports Evidently)      │
        │                                          │
        │  Auto-refresh : 30 secondes              │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │      Détection Data Drift                │
        │                                          │
        │  • DriftDetector (Evidently AI 0.7.17)   │
        │  • Rapports HTML/JSON (reports/drift/)   │
        │  • Seuil d'alerte : 30% features         │
        │  • Période : 7 jours par défaut          │
        └──────────────────────────────────────────┘
```

### Composants

| Composant | Technologie | Port | Rôle |
|-----------|-------------|------|------|
| **API** | FastAPI + Uvicorn | 8000 | Endpoints de prédiction |
| **Base de données** | PostgreSQL 16 | 5432 | Stockage persistant |
| **Dashboard** | Streamlit | 8501 | Visualisation temps réel |
| **Drift Detection** | Evidently AI | - | Analyse distribution |

---

## 🚀 Lancer le Dashboard

### Prérequis

1. **PostgreSQL actif** :
   ```bash
   brew services list
   # Si inactif :
   brew services start postgresql@16
   ```

2. **Base de données initialisée** :
   ```bash
   psql -U moon -d credit_scoring_prod -c "\dt"
   # Devrait afficher : predictions, feature_values, anomalies, drift_reports
   ```

3. **Environnement Python activé** :
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # ou
   .venv\Scripts\activate     # Windows
   ```

### Lancement

#### Option 1 : Dashboard seul
```bash
streamlit run src/monitoring/dashboard.py --server.port 8501
```

**Accès** : http://localhost:8501

#### Option 2 : Dashboard + API (2 terminaux)
```bash
# Terminal 1 : API
ENVIRONMENT=local LOG_LEVEL=INFO uvicorn src.api.main:app --reload --port 8000

# Terminal 2 : Dashboard
streamlit run src/monitoring/dashboard.py --server.port 8501
```

#### Option 3 : Avec docker-compose
```bash
docker-compose up -d
```
- API : http://localhost:8000
- Dashboard : http://localhost:8501

### Arrêter le dashboard

- **Mode interactif** : `Ctrl + C`
- **Processus background** :
  ```bash
  pkill -f "streamlit run"
  ```

---

## 📊 Interpréter les Métriques

### Page 🏠 Accueil

**Statut Système**

| Indicateur | Signification |
|------------|---------------|
| ✅ API opérationnelle | FastAPI répond (status 200) |
| ❌ API inaccessible | Vérifier `uvicorn` actif sur port 8000 |
| ✅ Base de données connectée | PostgreSQL accessible |
| ❌ PostgreSQL inaccessible | Vérifier `brew services list` |

**Métriques clés** :
- **Prédictions totales** : Nombre total en base (table `predictions`)
- **Taux d'approbation** : % de clients approuvés (decision='approve')

---

### Page 📈 Overview

**4 KPIs principaux**

1. **Total Prédictions**
   - Nombre de requêtes traitées
   - Filtrable par période (24h, 7j, 30j, Tout)

2. **Taux d'Approbation**
   - Formule : `(approve_count / total_predictions) × 100`
   - **Normal** : 70-85% (selon population)
   - **⚠️ Alerte** : < 50% ou > 95% (anomalie possible)

3. **Latence Moyenne**
   - Temps de réponse total (ms)
   - **Bon** : < 50 ms
   - **Acceptable** : 50-100 ms
   - **⚠️ Lent** : > 100 ms

4. **Taux d'Erreur**
   - Formule : `(error_count / total_predictions) × 100`
   - **Normal** : < 1%
   - **⚠️ Alerte** : > 5%

**Graphiques**

- **Donut Chart** : Répartition Approve/Refuse
  - Vérifie équilibre des décisions

- **Line Chart** : Volume par heure
  - Détecte pics de trafic
  - Identifie heures creuses

- **Histogram** : Distribution probabilités
  - Ligne rouge = seuil de décision (0.5225)
  - Concentration à gauche (< 0.5) = population à faible risque
  - Concentration à droite (> 0.5) = population à haut risque

- **Bar Chart** : Niveaux de confiance
  - **LOW** : Probabilité proche du seuil (0.4-0.6)
  - **MEDIUM** : Probabilité modérée (0.3-0.4 ou 0.6-0.7)
  - **HIGH** : Probabilité éloignée du seuil (< 0.3 ou > 0.7)

---

### Page ⚡ Performance

**Boxplot Latences**

- **Médiane** (ligne centrale) : temps typique
- **Q1-Q3** (boîte) : 50% des requêtes
- **Outliers** (points) : requêtes anormalement lentes

**Interprétation** :
- Médiane < 50 ms : ✅ Excellent
- Médiane 50-100 ms : ⚠️ Acceptable
- Médiane > 100 ms : ❌ Investigation nécessaire

**Top 10 Requêtes Lentes**

Action si latence > 200 ms :
1. Vérifier charge CPU/mémoire
2. Profiler avec cProfile (Étape 4)
3. Optimiser preprocessing si nécessaire

**Erreurs HTTP**

- **Code 200** : Succès
- **Code 422** : Validation échouée (données invalides)
- **Code 500** : Erreur interne (bug modèle/code)

---

### Page 💼 Business

**Pie Chart Profils**

- Rapport approve/refuse
- Vérifie alignement avec objectifs métier

**Histogram Montants**

- Distribution des crédits demandés
- Superposition par décision (approve en vert, refuse en rouge)

**Interprétation** :
- Montants élevés refusés : ✅ Normal (risque++)
- Petits montants refusés : ⚠️ Vérifier seuil trop strict

---

### Page 🔍 Data Drift

**KPIs Drift**

1. **Drift Détecté**
   - ✅ NON : Distribution stable
   - ⚠️ OUI : Changement significatif détecté

2. **Score de Drift**
   - Formule : `n_features_drifted / n_features_analyzed`
   - **Normal** : < 0.2 (20%)
   - **⚠️ Attention** : 0.2-0.3 (20-30%)
   - **❌ Critique** : > 0.3 (30%) → **Réentraînement recommandé**

3. **Features Affectées**
   - Nombre de features avec drift significatif
   - Liste détaillée si > 0

**Rapport Evidently HTML**

- Iframe interactive (scrollable)
- Tests statistiques par feature :
  - Kolmogorov-Smirnov (variables numériques)
  - Chi-carré (variables catégorielles)
- Visualisations : distributions, histogrammes

**Historique des Scores**

- Line chart avec évolution dans le temps
- Ligne rouge = seuil d'alerte (0.3)
- Tendance à la hausse = **alerte précoce**

---

## 🔔 Configuration des Alertes

### Seuils Recommandés

```python
# src/monitoring/drift_detector.py
DRIFT_THRESHOLD = 0.3  # 30% des features

# src/monitoring/storage.py
ERROR_RATE_THRESHOLD = 0.05  # 5%
LATENCY_THRESHOLD_MS = 100
```

### Script de vérification périodique

**Créer un fichier `check_alerts.py` (optionnel)** :

```python
"""
Script de vérification des alertes (à lancer en cron).

Usage:
    python src/scripts/check_alerts.py
"""
from src.monitoring.storage import PredictionStorage
from src.api.config import settings
from datetime import datetime, timedelta

def check_alerts():
    storage = PredictionStorage(database_url=settings.database_url)

    # Stats dernières 24h
    stats = storage.get_stats(start_date=datetime.now() - timedelta(hours=24))

    alerts = []

    # Alerte taux d'erreur
    error_rate = stats['error_count'] / stats['total_predictions'] if stats['total_predictions'] > 0 else 0
    if error_rate > 0.05:
        alerts.append(f"⚠️ Taux d'erreur élevé: {error_rate:.2%}")

    # Alerte latence
    if stats['avg_inference_time_ms'] > 100:
        alerts.append(f"⚠️ Latence élevée: {stats['avg_inference_time_ms']:.2f} ms")

    # Afficher alertes
    if alerts:
        print("=" * 60)
        print("ALERTES DÉTECTÉES")
        print("=" * 60)
        for alert in alerts:
            print(alert)
    else:
        print("✅ Aucune alerte")

    storage.close()

if __name__ == "__main__":
    check_alerts()
```

### Configuration cron (macOS/Linux)

```bash
# Éditer crontab
crontab -e

# Ajouter ligne (vérification toutes les heures)
0 * * * * cd /Users/mounirmeknaci/Desktop/Data_Projects/Projet8 && source .venv/bin/activate && python src/scripts/check_alerts.py >> logs/alerts.log 2>&1
```

### Notifications (exemples)

**Slack** :
```python
import requests

def send_slack_alert(message):
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    requests.post(webhook_url, json={"text": message})
```

**Email** :
```python
import smtplib
from email.message import EmailMessage

def send_email_alert(subject, body):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = "monitoring@example.com"
    msg['To'] = "team@example.com"
    msg.set_content(body)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login("user", "password")
        server.send_message(msg)
```

---

## 🛠️ Troubleshooting

### Problème 1 : Dashboard n'affiche aucune donnée

**Symptômes** :
- Message "Aucune prédiction disponible"
- KPIs à zéro

**Solutions** :

1. **Vérifier données en base** :
   ```bash
   psql -U moon -d credit_scoring_prod -c "SELECT COUNT(*) FROM predictions;"
   ```

   Si 0 lignes :
   ```bash
   # Générer du trafic
   python src/scripts/simulate_traffic.py --num-predictions 50
   ```

2. **Vérifier connexion PostgreSQL** :
   ```bash
   psql -U moon -d credit_scoring_prod -c "SELECT 1;"
   ```

   Si erreur :
   ```bash
   brew services restart postgresql@16
   ```

---

### Problème 2 : API inaccessible

**Symptômes** :
- Page d'accueil affiche "❌ API inaccessible"
- Erreur "Connection refused"

**Solutions** :

1. **Vérifier API lancée** :
   ```bash
   lsof -ti:8000
   # Si vide, lancer l'API
   uvicorn src.api.main:app --reload --port 8000
   ```

2. **Tester manuellement** :
   ```bash
   curl http://localhost:8000/
   ```

---

### Problème 3 : Erreur lors génération rapport drift

**Symptômes** :
- `ValueError: An empty column 'X' was provided`

**Solution** :
- Déjà corrigé dans `drift_detector.py` (filtrage automatique colonnes vides)
- Si persiste, vérifier version Evidently :
  ```bash
  pip show evidently
  # Devrait être 0.7.17
  ```

---

### Problème 4 : Streamlit se bloque

**Solutions** :

1. **Vider cache Streamlit** :
   ```bash
   streamlit cache clear
   ```

2. **Redémarrer** :
   ```bash
   pkill -f "streamlit run"
   streamlit run src/monitoring/dashboard.py --server.port 8501
   ```

---

### Problème 5 : PostgreSQL "too many connections"

**Solutions** :

1. **Vérifier pool de connexions** :
   ```python
   # src/monitoring/storage.py
   pool_size=5,
   max_overflow=10
   # Total max : 15 connexions
   ```

2. **Fermer connexions orphelines** :
   ```sql
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE datname = 'credit_scoring_prod'
   AND pid <> pg_backend_pid();
   ```

---

### Problème 6 : Rapport drift "DatasetDriftMetric non trouvé"

**Cause** : Evidently AI n'a pas trouvé assez de données pour calculer le drift

**Solutions** :

1. **Vérifier taille dataset production** :
   ```bash
   psql -U moon -d credit_scoring_prod -c "SELECT COUNT(*) FROM predictions WHERE timestamp >= NOW() - INTERVAL '7 days';"
   ```

   Minimum recommandé : **50+ prédictions**

2. **Augmenter période** :
   ```bash
   python src/scripts/generate_drift_report.py --days 30
   ```

---

## 📚 Références

### Commandes Utiles

```bash
# PostgreSQL
psql -U moon -d credit_scoring_prod                      # Connexion
\dt                                                       # Liste tables
SELECT COUNT(*) FROM predictions;                         # Compter prédictions
SELECT decision, COUNT(*) FROM predictions GROUP BY decision;  # Stats décisions

# Streamlit
streamlit run src/monitoring/dashboard.py --server.port 8501  # Lancer
streamlit cache clear                                     # Vider cache
pkill -f "streamlit run"                                  # Arrêter

# Drift
python src/scripts/generate_drift_report.py --days 7 --threshold 0.3  # Générer rapport
ls -lh reports/drift/                                     # Lister rapports

# Simulation
python src/scripts/simulate_traffic.py --num-predictions 100  # Générer trafic
```

### Documentation Externe

- **Evidently AI** : https://docs.evidentlyai.com/
- **Streamlit** : https://docs.streamlit.io/
- **PostgreSQL** : https://www.postgresql.org/docs/16/
- **FastAPI** : https://fastapi.tiangolo.com/

---

**Dernière mise à jour** : 15 décembre 2025
**Auteur** : Mounir Meknaci
**Projet** : Home Credit - API de Scoring Crédit (MLOps)
