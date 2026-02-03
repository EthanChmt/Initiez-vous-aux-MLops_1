import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    path = os.path.join(BASE_DIR, 'data', 'processed')
    X = np.load(os.path.join(path, 'train_final.npy'))
    y = np.load(os.path.join(path, 'train_labels.npy'))
    return X, y

def custom_business_cost(y_true, y_pred_proba, threshold=0.5):
    """Calcule un coût métier : un Faux Négatif coûte 10x plus qu'un Faux Positif."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Le coût : 10 pour un mauvais client non détecté, 1 pour un bon client refusé
    cost = (fn * 10) + (fp * 1)
    return cost

def train_and_log_robust(model_name, model, X, y):
    with mlflow.start_run(run_name=f"{model_name}_CV_Robust"):
        # 1. Pipeline complet
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # 2. Validation Croisée Stratifiée (Robustesse)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Évaluation robuste de {model_name} via Cross-Validation...")
        
        # On récupère les probabilités hors-échantillon (Out-of-fold)
        y_proba = cross_val_predict(pipeline, X, y, cv=skf, method='predict_proba')[:, 1]
        
        # 3. Calcul des métriques demandées
        auc_score = roc_auc_score(y, y_proba)
        recall_min = recall_score(y, (y_proba >= 0.5).astype(int))
        f1 = f1_score(y, (y_proba >= 0.5).astype(int))
        biz_cost = custom_business_cost(y, y_proba)

        # 4. Enregistrement MLflow
        mlflow.log_params(model.get_params() if hasattr(model, 'get_params') else {})
        mlflow.log_metric("val_roc_auc", auc_score)
        mlflow.log_metric("recall_minority", recall_min)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("business_cost", biz_cost)

        # Courbe ROC
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        plt.title(f'ROC Curve CV - {model_name}')
        plt.legend()
        plt.savefig("roc_cv.png")
        mlflow.log_artifact("roc_cv.png")
        plt.close()

        # 5. Entraînement final sur tout le dataset pour le Registry
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, "model", registered_model_name="HomeCredit_Final")
        
        print(f"{model_name} -> AUC: {auc_score:.4f} | Recall: {recall_min:.4f} | Cost: {biz_cost}")

def run_all_experiments():
    X, y = load_data()
    mlflow.set_experiment("Home_Credit_Robust_Evaluation")

    models = {
        "LogReg": LogisticRegression(C=0.1, class_weight='balanced', max_iter=500),
        "LGBM": LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGB": XGBClassifier(n_estimators=100, scale_pos_weight=11, random_state=42)
    }

    for name, obj in models.items():
        train_and_log_robust(name, obj, X, y)

if __name__ == "__main__":
    run_all_experiments()