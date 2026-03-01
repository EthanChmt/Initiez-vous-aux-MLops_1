import os
import logging
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (make_scorer, roc_auc_score, roc_curve, auc, 
                             balanced_accuracy_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from model_utils import load_data, custom_business_cost

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def get_model_config(model_type):
    if model_type == "Logistic_Regression":
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ])
        # Grille élargie
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l2']
        }
        
    elif model_type == "XGBoost":
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', XGBClassifier(scale_pos_weight=11, random_state=42, eval_metric='logloss'))
        ])
        param_grid = {
            'classifier__max_depth': [3, 5, 7],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1]
        }

    elif model_type == "LightGBM":
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', LGBMClassifier(class_weight='balanced', random_state=42, force_col_wise=True))
        ])
        param_grid = {
            'classifier__num_leaves': [20, 31, 50],
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1]
        }
    
    return pipeline, param_grid

def find_best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.1, 0.9, 81)
    costs = [custom_business_cost(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmin(costs)]
    return best_threshold, np.min(costs)

def run_exhaustive_tuning(sample_size=50000):
    X_full, y_full = load_data()
    X, _, y, _ = train_test_split(X_full, y_full, train_size=sample_size, stratify=y_full, random_state=42) 
    
    biz_scorer = make_scorer(custom_business_cost, greater_is_better=False, needs_proba=True)
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    model_types = ["Logistic_Regression", "XGBoost", "LightGBM"]
    mlflow.set_experiment("Credit_Scoring_Optimized")

    for m_type in model_types:
        pipeline, param_grid = get_model_config(m_type)
        
        # Le run "Parent"
        with mlflow.start_run(run_name=f"Parent_{m_type}"):
            
            # Activation de l'autolog AVANT le fit
            # log_input_examples=True aide pour le déploiement futur
            mlflow.sklearn.autolog(
                log_models=True, 
                log_datasets=True, 
                disable=False,
                exclusive=False
            ) 
            
            grid = GridSearchCV(
                pipeline, param_grid, cv=cv_strategy, 
                scoring={'AUC': 'roc_auc', 'Cost': biz_scorer},
                refit='Cost', n_jobs=-1
            )
            
            # Cette étape va créer automatiquement des "Child Runs" pour chaque combo de param_grid
            grid.fit(X, y)

            best_model = grid.best_estimator_ 
            y_proba = best_model.predict_proba(X)[:, 1]
            best_thresh, min_cost = find_best_threshold(y, y_proba)
            y_pred_optimal = (y_proba >= best_thresh).astype(int)

            # Log des métriques de l'étape d'optimisation de seuil sur le parent
            mlflow.log_metrics({
                "optimized_threshold": best_thresh,
                "min_business_cost": min_cost,
                "final_balanced_accuracy": balanced_accuracy_score(y, y_pred_optimal)
            })

            # Artifacts visuels
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y, y_pred_optimal, cmap='Blues', ax=ax)
            ax.set_title(f"Confusion Matrix (Thresh: {best_thresh:.2f})")
            cm_path = f"cm_{m_type}.png"
            plt.savefig(cm_path); mlflow.log_artifact(cm_path); plt.close()

            # --- MODEL REGISTRY ---
            # On enregistre le meilleur modèle dans le registre centralisé
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, f"Credit_Model_{m_type}")
            
            if os.path.exists(cm_path): os.remove(cm_path)
            
            logger.info(f"Modèle {m_type} enregistré dans le Registry.")

if __name__ == "__main__":
    run_exhaustive_tuning()