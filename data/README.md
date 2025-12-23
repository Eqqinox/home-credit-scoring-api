# Données du projet

## Fichier principal

**`app_train_models.csv`**

### Description

Données préparées pour la modélisation :
- **307,511 clients**
- **646 features** (après feature engineering et agrégation)
- **Target** : défaut de paiement (0 = pas de défaut, 1 = défaut)
- **Déséquilibre des classes** : 91.9% classe 0 / 8.1% classe 1

### Provenance

Les données proviennent du Kaggle Competition [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data).

Elles ont été préparées lors de la **Partie 1 du projet** :
- Nettoyage et exploration (EDA)
- Agrégation des tables secondaires (bureau, previous_application, etc.)
- Feature engineering
- Encodage préparé pour la modélisation

### Traitement appliqué

1. **Nettoyage** : Gestion des valeurs manquantes
2. **Agrégation** : Fusion des tables secondaires
3. **Feature Engineering** : Création de 646 features
4. **Préparation** : Prêt pour l'encodage et la modélisation

### Note importante

Ce fichier **N'EST PAS versionné** dans Git car trop volumineux (plusieurs centaines de Mo).

### Comment obtenir le fichier

#### Depuis Kaggle (données brutes + preprocessing)
1. Télécharger les données depuis Kaggle
2. Appliquer le notebook d'exploration (Initiez-vous au MLOps (partie 1/2)- Openclassrooms)
3. Générer le fichier `app_train_models.csv`

### Structure attendue
```
data/
├── README.md                    # Ce fichier
└── app_train_models.csv         # Fichier de données (non versionné)
```

---

*Dernière mise à jour: Décembre 2025*  
*Projet Home Credit Scoring API - OpenClassrooms*.  
*Auteur : Mounir Meknaci*.  
*Version : 1.0*