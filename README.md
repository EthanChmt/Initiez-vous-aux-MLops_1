# Projet de Scoring Crédit - Home Credit Default Risk

Ce projet vise à développer un modèle de scoring pour prédire la probabilité de défaut de paiement des clients de **Home Credit**. L'objectif est de minimiser le coût métier total en optimisant le seuil de décision entre les bons et les mauvais payeurs.

## 📊 Données
Les données utilisées proviennent de la compétition Kaggle : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data).

### Installation des données
1. Telechargez les fichiers CSV depuis le lien Kaggle ci-dessus.
2. Créez un dossier nommé `data/` à la racine du projet.
3. Placez-y les fichiers, notamment `application_train.csv`.
4. Notez que le dossier `data/` est exclu du suivi Git via le fichier `.gitignore`.

---

## 🛠️ Structure du Projet

```text
├── data/                   # Données brutes (à télécharger)
├── notebooks/              
│   └── Analyse_exploratoire.ipynb  # Notebooks de nettoyage et préparation des données
├── src/                    
│   ├── model_utils.py      # Fonctions utilitaires (chargement, coût métier personnalisé)
│   ├── train_all.py        # SCRIPT PRINCIPAL : Tracking MLflow et optimisation du seuil
│   ├── train.py            # (Legacy) Ancien script d'entraînement simple
│   └── tunning.py          # (Legacy) Ancien script d'optimisation (vieillissant)
├── mlflow.db               # Base de données SQLite pour le tracking MLflow
├── pyproject.toml          # Configuration des dépendances (Poetry)
└── README.md
```

## 🚀 Utilisation

### 1. Préparation des données
Utilisez le notebook `notebooks/Analyse_exploratoire.ipynb` pour effectuer le nettoyage des données. Ce notebook permet de traiter les valeurs manquantes, d'encoder les variables et de préparer le dataset final pour l'entraînement.

### 2. Entraînement et Tracking avec MLflow
Le script `src/train_all.py` est le script de référence pour la modélisation.

**Fonctionnalités clés :**
* **Connexion à MLflow** : Le script est configuré pour enregistrer les runs dans `sqlite:///mlflow.db`.
* **Optimisation du Seuil** : Recherche automatique du meilleur seuil de probabilité pour minimiser le coût métier (FN > FP).
* **Tracking Complet** : Enregistrement des métriques (AUC, Balanced Accuracy, Coût), des hyperparamètres et des artefacts (Matrices de confusion, courbes ROC).

Pour lancer l'entraînement complet :
```bash
python src/train_all.py
```
### 3. Visualisation des résultats
Pour accéder à l'interface de visualisation MLflow et comparer les différents modèles entraînés :

```bash
mlflow ui
```
L'interface sera disponible à l'adresse http://127.0.0.1:5000.

## ⚖️ Logique Métier
L'évaluation des modèles repose sur une fonction de coût personnalisée (définie dans model_utils.py). Cette approche pénalise plus fortement les Faux Négatifs (prêts accordés à tort) que les Faux Positifs, assurant une performance alignée avec les enjeux financiers de l'octroi de crédit.

Cette fonction est intégrée directement dans le processus d'optimisation via un make_scorer, permettant de sélectionner les hyperparamètres et le seuil de probabilité qui minimisent la perte financière totale plutôt que de simplement maximiser l'accuracy statistique.
