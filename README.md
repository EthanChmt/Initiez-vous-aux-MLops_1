# 🏦 Projet de Scoring Crédit — Home Credit Default Risk

> Modèle de machine learning pour prédire la probabilité de défaut de paiement, aligné sur les enjeux financiers réels de l'octroi de crédit.

---

## 📌 Contexte

Ce projet s'inscrit dans le cadre d'une approche **MLOps** appliquée au scoring crédit pour **Home Credit**. L'objectif est de construire, tracker et optimiser un modèle de classification binaire qui minimise le **coût métier total** — en pénalisant davantage les Faux Négatifs (prêts accordés à des clients défaillants) que les Faux Positifs.

Les données proviennent de la compétition Kaggle : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data).

---

## 🗂️ Structure du Projet

```
├── data/
│   ├── raw/                            # Données brutes (à télécharger — non versionnées)
│   └── processed/                      # Données nettoyées et prêtes à l'entraînement
├── notebooks/
│   ├── Analyse_exploratoire.ipynb      # Nettoyage, encodage et préparation des données
│   └── training_experimentation.ipynb  # Expérimentations et comparaison des modèles
├── reports/
│   └── figures/                        # Matrices de confusion et courbes ROC générées
│       ├── cm_LightGBM.png
│       ├── cm_Logistic_Regression.png
│       ├── cm_XGBoost.png
│       ├── roc_LightGBM.png
│       ├── roc_Logistic_Regression.png
│       └── roc_XGBoost.png
├── src/
│   ├── model_predict.py                # Inférence et prédiction sur nouvelles données
│   ├── model_utils.py                  # Chargement des données, scoring métier personnalisé
│   └── pipeline_threshold.py           # Pipeline d'entraînement + optimisation du seuil
├── mlruns/                             # Artefacts MLflow
├── mlflow.db                           # Base SQLite pour le tracking MLflow
├── pyproject.toml                      # Dépendances gérées avec Poetry
└── README.md
```

---

## ⚙️ Installation

Ce projet utilise [Poetry](https://python-poetry.org/) pour la gestion des dépendances.

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd <nom-du-projet>

# Installer les dépendances
poetry install

# Activer l'environnement virtuel
poetry shell
```

**Prérequis :** Python `^3.11.9`

### Données

1. Téléchargez les fichiers CSV depuis [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data).
2. Placez les fichiers bruts dans `data/raw/`, notamment `application_train.csv`.

> Le dossier `data/` est exclu du suivi Git via `.gitignore`.

---

## 🚀 Utilisation

### 1. Préparation des données

Lancez le notebook d'exploration pour nettoyer et préparer le dataset :

```bash
jupyter notebook notebooks/Analyse_exploratoire.ipynb
```

### 2. Entraînement et Tracking MLflow

Le script `src/pipeline_threshold.py` gère l'entraînement complet et l'optimisation du seuil de décision.

```bash
python src/pipeline_threshold.py
```

Ce script :
- Entraîne les modèles (LightGBM, XGBoost, Logistic Regression)
- Optimise automatiquement le **seuil de décision** pour minimiser le coût métier
- Enregistre dans MLflow : métriques, hyperparamètres, matrices de confusion, courbes ROC

Le tout premet d'être lancé par le notebook `notebooks/training_experiment.ipynb`
### 3. Visualisation des résultats

```bash
mlflow ui
```

Interface disponible sur : [http://127.0.0.1:5000](http://127.0.0.1:5000)
### 4. Prédiction

```bash
python src/model_predict.py
```


---

## ⚖️ Logique Métier

Le modèle est évalué via une **fonction de coût personnalisée** (définie dans `model_utils.py`) :

| Cas | Description | Poids |
|-----|-------------|-------|
| Faux Négatif (FN) | Prêt accordé à un client défaillant | ⚠️ Élevé |
| Faux Positif (FP) | Prêt refusé à un bon client | Faible |

Cette logique est intégrée via un `make_scorer` scikit-learn, garantissant que le **seuil de probabilité final** minimise la perte financière réelle plutôt que la simple accuracy statistique.

---

## 📈 Métriques Suivies

- **AUC-ROC** — discrimination globale du modèle
- **Balanced Accuracy** — robustesse face au déséquilibre de classes
- **Coût métier** — métrique principale alignée sur les enjeux financiers

---

## 👤 Auteur

**Ethan Chaumeret** — [ethan.chaumeret@gmail.com](mailto:ethan.chaumeret@gmail.com)