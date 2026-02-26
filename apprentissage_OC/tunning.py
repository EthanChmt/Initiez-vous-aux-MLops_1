import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from lightgbm import LGBMClassifier
from model_utils import load_data, custom_business_cost
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_exhaustive_tuning(sample_size=50000, n_jobs=-1):
    
    X, y = load_data()
    
    # Sample data if specified
    if sample_size:
        X, y = X[:sample_size], y[:sample_size]
        logger.info(f"Using {sample_size} samples for tuning")
    
    # Log class distribution
    class_dist = np.bincount(y) / len(y)
    logger.info(f"Class distribution: {class_dist}")
    
    mlflow.set_experiment("LGBM_Hyperparameter_Tuning")

    # Custom business scorer
    biz_scorer = make_scorer(
        custom_business_cost, 
        greater_is_better=False, # plus bas le coût est mieux c'est
        needs_proba=True
    )

    with mlflow.start_run(run_name="GridSearch_LGBM_Full"):
        # Log hyperparameters and configuration
        mlflow.log_param("sample_size", sample_size or "full")
        mlflow.log_param("cv_folds", 3)
        mlflow.log_metric("class_0_ratio", class_dist[0])
        mlflow.log_metric("class_1_ratio", class_dist[1])
        
        # Enable autolog but disable model logging to avoid duplicates
        mlflow.sklearn.autolog(
            log_models=False,  # We'll log the best model manually
            log_input_examples=False,
            log_model_signatures=False,
            silent=True
        )

        # Pipeline with LightGBM-specific optimizations
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LGBMClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=n_jobs,
                verbose=-1,  # Suppress LightGBM warnings
                force_col_wise=True  # Faster on wide datasets
            ))
        ])

        # Expanded and more targeted parameter grid
        param_grid = {
            'classifier__num_leaves': [31, 50],             # On garde la valeur par défaut et une plus complexe
            'classifier__learning_rate': [0.1, 0.05],       # Deux vitesses d'apprentissage classiques
            'classifier__n_estimators': [100, 200],         # 100 pour la rapidité, 200 pour la performance
            'classifier__max_depth': [-1, 10],              # Illimité vs contraint
            'classifier__min_child_samples': [20],          # On fixe une valeur robuste
            'classifier__subsample': [0.8, 1.0],            # Avec et sans échantillonnage de lignes
            'classifier__colsample_bytree': [1.0]           # On reste sur l'intégralité des colonnes
        }

        # Use StratifiedKFold for better class balance
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv,
            scoring={'AUC': 'roc_auc', 'Cost': biz_scorer},
            refit='Cost',  # Optimize for business metric
            n_jobs=n_jobs,
            verbose=2,
            return_train_score=True,
            error_score='raise'
        )
        
        logger.info(f"Starting grid search with {len(param_grid['classifier__num_leaves']) * len(param_grid['classifier__learning_rate']) * len(param_grid['classifier__n_estimators']) * len(param_grid['classifier__max_depth']) * len(param_grid['classifier__min_child_samples']) * len(param_grid['classifier__subsample']) * len(param_grid['classifier__colsample_bytree'])} combinations")
        
        grid.fit(X, y)
        
        # Log best parameters and scores
        logger.info(f"Best parameters: {grid.best_params_}")
        logger.info(f"Best CV score (Cost): {grid.best_score_}")
        
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_cost", grid.best_score_)
        mlflow.log_metric("best_cv_auc", grid.cv_results_['mean_test_AUC'][grid.best_index_])
        
        # Log top 5 parameter combinations
        results_df = pd.DataFrame(grid.cv_results_)
        top_5 = results_df.nsmallest(5, 'rank_test_Cost')[
            ['params', 'mean_test_Cost', 'mean_test_AUC', 'std_test_Cost']
        ]
        mlflow.log_text(top_5.to_string(), "top_5_configurations.txt")
        
        # Register best model with metadata
        model_info = mlflow.sklearn.log_model(
            grid.best_estimator_, 
            "best_lgbm_model",
            registered_model_name="HomeCredit_Final_Tuned",
            signature=mlflow.models.infer_signature(X, grid.predict_proba(X)),
            input_example=X[:5]
        )
        
        logger.info(f"Model registered: {model_info.model_uri}")
        
        return grid.best_estimator_, grid.best_params_, grid.best_score_


if __name__ == "__main__":
    best_model, best_params, best_score = run_exhaustive_tuning(
        sample_size=50000,
        n_jobs=-1
    )