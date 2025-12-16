# 🔒 Conformité RGPD - Credit Scoring API

> Analyse de conformité au Règlement Général sur la Protection des Données (RGPD)

---

## 📋 Sommaire

1. [Contexte du projet](#contexte-du-projet)
2. [Données collectées](#données-collectées)
3. [Mesures de protection](#mesures-de-protection)
4. [Recommandations pour production réelle](#recommandations-pour-production-réelle)
5. [Limites du projet académique](#limites-du-projet-académique)

---

## 🎯 Contexte du projet

### Nature des données

Ce projet utilise le dataset **Kaggle Home Credit Default Risk** à des fins **académiques et pédagogiques**.

**Important** : Les données sont déjà **anonymisées par Kaggle** :
- `SK_ID_CURR` : Identifiant fictif généré (non lié à de vrais clients)
- Aucune donnée personnelle directement identifiante (nom, prénom, adresse, etc.)
- Variables agrégées et transformées

### Statut RGPD

| Critère | Statut | Justification |
|---------|--------|---------------|
| Données personnelles réelles | ❌ NON | Dataset Kaggle anonymisé |
| Identifiants directs | ❌ NON | SK_ID_CURR est fictif |
| Traitement à des fins commerciales | ❌ NON | Projet académique (OpenClassrooms) |
| Stockage données sensibles | ⚠️ PARTIEL | Variables financières (montants, revenus) mais non identifiantes |

**Conclusion** : Le projet n'est **pas directement soumis au RGPD** car il ne traite pas de données personnelles réelles. Cependant, cette documentation présente les bonnes pratiques à appliquer pour une **mise en production réelle**.

---

## 📊 Données collectées

### Base de données PostgreSQL

La base `credit_scoring_prod` stocke les informations suivantes :

#### Table `predictions` (15 colonnes)

| Colonne | Type | Données sensibles ? | Justification |
|---------|------|---------------------|---------------|
| `id` | Integer (PK) | ❌ NON | ID technique auto-incrémenté |
| `created_at` | Timestamp | ⚠️ MÉTADONNÉE | Date de traitement |
| `client_id` | Integer | ⚠️ IDENTIFIANT | **À pseudonymiser en production** |
| `probability` | Float | ✅ OUI | Score de risque (décision automatisée) |
| `prediction` | Integer | ✅ OUI | Résultat binaire (0/1) |
| `decision` | String | ✅ OUI | Décision métier (approve/refuse) |
| `confidence_level` | String | ⚠️ MÉTADONNÉE | Niveau de confiance (LOW/MEDIUM/HIGH) |
| `threshold_used` | Float | ❌ NON | Paramètre technique |
| `model_version` | String | ❌ NON | Version du modèle |
| `preprocessing_time_ms` | Float | ❌ NON | Métrique technique |
| `inference_time_ms` | Float | ❌ NON | Métrique technique |
| `total_time_ms` | Float | ❌ NON | Métrique technique |
| `http_status_code` | Integer | ❌ NON | Code réponse HTTP |
| `endpoint` | String | ❌ NON | Endpoint appelé |
| `data_quality_score` | Float | ⚠️ MÉTADONNÉE | Score qualité données |

#### Table `feature_values` (top 20 features par prédiction)

| Colonne | Données sensibles ? |
|---------|---------------------|
| `id` | ❌ NON (ID technique) |
| `prediction_id` | ❌ NON (Foreign Key) |
| `feature_name` | ❌ NON (Nom variable) |
| `feature_value` | ✅ OUI (Valeur financière) |
| `created_at` | ⚠️ MÉTADONNÉE |

**Exemples de features stockées** :
- `AMT_CREDIT` : Montant du crédit demandé
- `AMT_INCOME_TOTAL` : Revenu total
- `EXT_SOURCE_1/2/3` : Scores externes (agrégés)
- `DAYS_BIRTH` : Âge (en jours négatifs)

#### Table `drift_reports` (rapports Evidently AI)

| Colonne | Données sensibles ? |
|---------|---------------------|
| `id` | ❌ NON |
| `report_data` | ⚠️ AGRÉGÉES | Statistiques agrégées (pas de données individuelles) |
| `created_at` | ⚠️ MÉTADONNÉE |

#### Table `anomalies` (logs d'erreurs)

| Colonne | Données sensibles ? |
|---------|---------------------|
| `id` | ❌ NON |
| `error_type` | ❌ NON |
| `error_message` | ⚠️ PEUT CONTENIR | Peut contenir client_id dans le message |
| `stack_trace` | ❌ NON |

### Fichiers de logs (structlog)

**Format** : JSON structuré (production) ou coloré (local)

**Exemple de log** :
```json
{
  "event": "prediction",
  "client_id": 100001,
  "probability": 0.3521,
  "decision": "approve",
  "preprocessing_ms": 12.5,
  "inference_ms": 5.2,
  "total_ms": 35.8,
  "timestamp": "2025-12-15T14:30:00Z"
}
```

**⚠️ Contient** : `client_id` (à pseudonymiser en production)

---

## 🛡️ Mesures de protection

### Mesures actuellement implémentées

| Mesure | Statut | Description |
|--------|--------|-------------|
| **Chiffrement en transit** | ✅ PRODUCTION | HTTPS sur Hugging Face Spaces |
| **Authentification API** | ❌ NON | API publique (académique) |
| **Validation des entrées** | ✅ OUI | Pydantic schemas (FastAPI) |
| **Limitation de requêtes** | ❌ NON | Pas de rate limiting |
| **Logs d'accès** | ✅ OUI | structlog avec timestamps |
| **Anonymisation client_id** | ❌ NON | Stockage en clair (données fictives) |
| **Politique de rétention** | ❌ NON | Pas de purge automatique |

### Sécurité PostgreSQL

```python
# src/monitoring/storage.py
# Connection pooling sécurisé
engine = create_engine(
    database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True  # Vérification connexions
)
```

**✅ Bonne pratique** : Pool de connexions limité (15 max)

**⚠️ Amélioration possible** : Credentials en variable d'environnement (actuellement en dur dans config)

---

## 🚀 Recommandations pour production réelle

### 1. Pseudonymisation des identifiants (Art. 4.5 RGPD)

**Implémentation recommandée** :

```python
# src/utils/gdpr.py
import hashlib
import secrets
from functools import lru_cache

# Salt généré une seule fois et stocké de manière sécurisée
# (ex: AWS Secrets Manager, HashiCorp Vault)
SALT = secrets.token_hex(32)  # 64 caractères

@lru_cache(maxsize=10000)
def pseudonymize_client_id(client_id: int) -> str:
    """
    Pseudonymise un client_id avec SHA256 + salt.

    Conforme RGPD Art. 4.5 : Pseudonymisation
    - Le sujet reste identifiable de manière indirecte (via le salt)
    - Réduction du risque en cas de fuite de données

    Args:
        client_id: Identifiant client original (int)

    Returns:
        Hash SHA256 (64 caractères hexadécimaux)
    """
    data = f"{client_id}{SALT}".encode('utf-8')
    return hashlib.sha256(data).hexdigest()

# Exemple d'utilisation
# original: 100001
# pseudonymisé: "a3f2b8c1d9e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0"
```

**Application** :
- Pseudonymiser `client_id` **avant** stockage PostgreSQL
- Pseudonymiser `client_id` dans les logs structlog
- Conserver mapping original ↔ pseudonymisé dans table séparée (accès restreint)

---

### 2. Politique de rétention des données (Art. 5.1.e RGPD)

**Recommandation** : **90 jours maximum** pour les données de monitoring

**Implémentation** :

```python
# src/scripts/purge_old_predictions.py
from datetime import datetime, timedelta
from src.monitoring.storage import PredictionStorage

def purge_old_predictions(retention_days: int = 90):
    """
    Supprime les prédictions plus anciennes que retention_days.

    Conforme RGPD Art. 5.1.e : Limitation de conservation
    """
    storage = PredictionStorage()
    cutoff_date = datetime.now() - timedelta(days=retention_days)

    # Suppression en cascade (predictions + feature_values)
    deleted_count = storage.delete_predictions_before(cutoff_date)

    print(f"✅ {deleted_count} prédictions supprimées (> {retention_days} jours)")
    storage.close()

# Planifier avec cron (quotidien à 2h du matin)
# 0 2 * * * cd /path/to/project && python src/scripts/purge_old_predictions.py
```

---

### 3. Droit à l'oubli (Art. 17 RGPD)

**Endpoint recommandé** :

```python
# src/api/main.py
@app.delete("/client/{client_id}", status_code=204)
async def delete_client_data(client_id: int):
    """
    Supprime toutes les données associées à un client.

    Conforme RGPD Art. 17 : Droit à l'effacement

    Args:
        client_id: Identifiant du client

    Returns:
        204 No Content si succès
    """
    storage = PredictionStorage()

    # Suppression en cascade
    deleted_predictions = storage.delete_by_client_id(client_id)

    if deleted_predictions == 0:
        raise HTTPException(status_code=404, detail="Client not found")

    storage.close()

    logger.info("client_data_deleted", client_id=client_id, count=deleted_predictions)
    return Response(status_code=204)
```

---

### 4. Chiffrement des données au repos (Art. 32 RGPD)

**Recommandation** : Chiffrer les colonnes sensibles dans PostgreSQL

```sql
-- Option 1 : Chiffrement au niveau de PostgreSQL (pgcrypto)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Modifier la table pour stocker les valeurs chiffrées
ALTER TABLE predictions
ADD COLUMN client_id_encrypted BYTEA;

-- Insertion chiffrée (exemple)
INSERT INTO predictions (client_id_encrypted, ...)
VALUES (
    pgp_sym_encrypt('100001', 'secret_key_from_vault'),
    ...
);

-- Option 2 : Chiffrement au niveau applicatif (Python)
from cryptography.fernet import Fernet

def encrypt_sensitive_data(data: str, key: bytes) -> bytes:
    f = Fernet(key)
    return f.encrypt(data.encode())
```

---

### 5. Contrôle d'accès et authentification (Art. 32 RGPD)

**Recommandations** :

1. **API Key authentication** :
   ```python
   from fastapi.security import APIKeyHeader

   api_key_header = APIKeyHeader(name="X-API-Key")

   @app.post("/predict")
   async def predict(api_key: str = Depends(api_key_header)):
       if api_key not in VALID_API_KEYS:
           raise HTTPException(status_code=403, detail="Invalid API key")
       # ...
   ```

2. **Rate limiting** (ex: 100 requêtes/heure par IP)

3. **Logs d'audit** : Tracer qui accède à quelles données

---

### 6. Registre des traitements (Art. 30 RGPD)

**À documenter** :

| Élément | Description |
|---------|-------------|
| **Finalité** | Scoring crédit automatisé |
| **Base légale** | Consentement (Art. 6.1.a) ou Contrat (Art. 6.1.b) |
| **Catégories de données** | Données financières (revenus, crédits, scores externes) |
| **Destinataires** | Équipe Data Science, Service Crédit |
| **Durée de conservation** | 90 jours (monitoring), durée du contrat (dossier client) |
| **Mesures de sécurité** | Chiffrement HTTPS, pseudonymisation, accès restreint |
| **Transferts hors UE** | Non applicable (hébergement UE) |

---

## ⚠️ Limites du projet académique

### Écarts RGPD (justifiés par le contexte)

| Écart | Justification | En production |
|-------|---------------|---------------|
| Pas de pseudonymisation | Données Kaggle fictives | ⚠️ **OBLIGATOIRE** |
| Pas d'authentification API | Démonstration publique | ⚠️ **OBLIGATOIRE** |
| Pas de politique de rétention | Besoin de données pour tests | ⚠️ **OBLIGATOIRE** |
| Pas de chiffrement au repos | Simplicité architecture académique | ✅ Recommandé |
| Logs contiennent client_id | Debugging facilité | ⚠️ **À PSEUDONYMISER** |

### Déclaration CNIL

**En production** : Déclarer le traitement auprès de la CNIL si :
- Décision automatisée produisant des effets juridiques (refus de crédit)
- Scoring (Art. 22 RGPD)

**Mesures compensatoires** (Art. 22.3) :
- Droit à intervention humaine
- Droit d'exprimer son point de vue
- Droit de contester la décision

---

## 📚 Références

### Textes légaux

- **RGPD** : Règlement (UE) 2016/679 du 27 avril 2016
- **Loi Informatique et Libertés** : Loi n° 78-17 du 6 janvier 1978 (modifiée)

### Articles clés

| Article | Sujet | Application |
|---------|-------|-------------|
| Art. 4.5 | Pseudonymisation | Hachage client_id |
| Art. 5.1.e | Limitation de conservation | Purge 90 jours |
| Art. 17 | Droit à l'effacement | Endpoint DELETE |
| Art. 22 | Décision automatisée | Scoring crédit |
| Art. 30 | Registre des traitements | Documentation |
| Art. 32 | Sécurité du traitement | Chiffrement, accès |

### Ressources

- **CNIL** : https://www.cnil.fr/fr/reglement-europeen-protection-donnees
- **Guide CNIL** : https://www.cnil.fr/fr/guide-de-la-securite-des-donnees-personnelles

---

## ✅ Checklist RGPD Production

Avant mise en production réelle :

- [ ] Pseudonymiser tous les `client_id` (SHA256 + salt)
- [ ] Implémenter authentification API (API keys)
- [ ] Activer chiffrement HTTPS (certificat SSL/TLS)
- [ ] Configurer politique de rétention (90 jours max)
- [ ] Créer endpoint droit à l'oubli (`DELETE /client/{id}`)
- [ ] Chiffrer données sensibles au repos (pgcrypto ou Fernet)
- [ ] Mettre en place rate limiting (prévention abus)
- [ ] Documenter registre des traitements (Art. 30)
- [ ] Obtenir consentement explicite des clients (si applicable)
- [ ] Former les équipes au RGPD
- [ ] Désigner un DPO (Data Protection Officer) si > 250 employés
- [ ] Effectuer AIPD (Analyse d'Impact) si risque élevé

---

**Dernière mise à jour** : 15 décembre 2025
**Auteur** : Mounir Meknaci
**Statut** : Documentation académique (non applicable en l'état)
**Version** : 1.0
