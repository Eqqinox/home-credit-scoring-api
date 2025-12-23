# Guide d'Utilisation de l'API Credit Scoring

## URL de l'API Déployée

- **Production**: https://eqqinox-credit-scoring-api.hf.space
- **Swagger UI**: https://eqqinox-credit-scoring-api.hf.space/docs

---

## Endpoints Disponibles

### 1. Health Check
```bash
curl https://eqqinox-credit-scoring-api.hf.space/
```

### 2. Informations sur le Modèle
```bash
curl https://eqqinox-credit-scoring-api.hf.space/model-info
```

### 3. Prédiction Unique
```bash
curl -X POST https://eqqinox-credit-scoring-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d @example_single_request.json
```

### 4. Prédiction en Batch
```bash
curl -X POST https://eqqinox-credit-scoring-api.hf.space/predict-batch \
  -H "Content-Type: application/json" \
  -d @example_batch_request.json
```

---

## Fichiers d'Exemple

### `example_single_request.json`
- Contient **1 client** avec toutes les 645 features
- Utilisé pour tester `/predict`
- Client ID: 100002

### `example_batch_request.json`
- Contient **3 clients** avec toutes les features
- Utilisé pour tester `/predict-batch`
- Client IDs: 100002, 100003, 100004

---

## Tests Locaux

### Test en local (port 8000)
```bash
# Prédiction unique
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @example_single_request.json

# Prédiction batch
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d @example_batch_request.json
```

---

## Réponse Attendue

### `/predict` (single)
```json
{
  "client_id": 100002,
  "probability_default": 0.0823,
  "prediction": 0,
  "decision": "approve",
  "risk_level": "low",
  "threshold_used": 0.5225,
  "model_version": "v1.0",
  "timestamp": "2025-12-08T12:00:00"
}
```

### `/predict-batch`
```json
{
  "predictions": [
    {
      "client_id": 100002,
      "probability_default": 0.0823,
      "prediction": 0,
      "decision": "approve",
      "risk_level": "low"
    },
    {
      "client_id": 100003,
      "probability_default": 0.1245,
      "prediction": 0,
      "decision": "approve",
      "risk_level": "low"
    },
    {
      "client_id": 100004,
      "probability_default": 0.6531,
      "prediction": 1,
      "decision": "refuse",
      "risk_level": "high"
    }
  ],
  "count": 3,
  "timestamp": "2025-12-08T12:00:00"
}
```

---

## Notes Importantes

### Features Requises
L'API nécessite **toutes les 645 features** du dataset d'entraînement :
- Features de base (AMT_INCOME_TOTAL, AMT_CREDIT, etc.)
- Features engineered (BUREAU_*, CC_*, PREV_*, INST_*, POS_*)
- Features catégorielles encodées

### Pourquoi autant de features ?
Le modèle a été entraîné avec un feature engineering complet incluant :
- Agrégations depuis bureau.csv (97 features)
- Agrégations depuis previous_application.csv (164 features)
- Agrégations depuis credit_card_balance.csv (149 features)
- Agrégations depuis installments_payments.csv (57 features)
- Agrégations depuis POS_CASH_balance.csv (44 features)

En production réelle, ces features seraient pré-calculées dans un data warehouse avant l'appel API.

---

## Dépannage

### Erreur 400: Features manquantes
```json
{
  "error": "Features manquantes: ['BUREAU_AMT_CREDIT_SUM_DEBT_MAX', ...]"
}
```
**Solution**: Utilisez les fichiers d'exemple fournis qui contiennent toutes les features.

### Erreur 422: Validation échouée
```json
{
  "detail": [
    {
      "loc": ["body", "AMT_INCOME_TOTAL"],
      "msg": "Revenu invalide",
      "type": "value_error"
    }
  ]
}
```
**Solution**: Vérifiez que les valeurs respectent les contraintes (revenu > 0, etc.)

---

*Dernière mise à jour: Décembre 2025*  
*Projet Home Credit Scoring API - OpenClassrooms*.  
*Auteur : Mounir Meknaci*.  
*Version : 1.0*