import mlflow.pyfunc
import numpy as np
import logging
import warnings

# On importe load_data depuis ton module source
from src.model_utils import load_data

# Configuration des logs pour une sortie propre
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def make_predictions_with_champions():
    """
    Charge les 3 champions depuis le Model Registry et effectue des prédictions.
    """
    try:
        # 1. Chargement des données (Numpy arrays)
        X_full, _ = load_data()
        
        # CORRECTION : On utilise le slicing [0:5] car X_full est un numpy.ndarray
        sample_data = X_full[0:5]
        logger.info(f"Données de test chargées. Taille du sample : {sample_data.shape}")

        # 2. Liste de tes champions enregistrés dans le Registry
        # Assure-toi que ces noms correspondent exactement à ceux de ton script d'entraînement
        champions = [
            "Credit_Model_Logistic_Regression",
            "Credit_Model_XGBoost",
            "Credit_Model_LightGBM"
        ]
        
        for model_name in champions:
            print("-" * 50)
            try:
                # On vise la version 1 ou le stage "None" (par défaut après registration)
                model_uri = f"models:/{model_name}/1"
                logger.info(f"Chargement du modèle : {model_name} (v1)...")
                
                # Chargement du modèle via l'API pyfunc (générique)
                model = mlflow.pyfunc.load_model(model_uri)
                
                # Prédiction
                preds = model.predict(sample_data)
                
                print(f"RÉSULTAT pour {model_name}:")
                print(f" -> Prédictions de classes : {preds}")
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {model_name} : {e}")
                print("Conseil : Vérifie dans l'UI MLflow que le modèle est bien enregistré avec ce nom exact.")

    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {e}")

if __name__ == "__main__":
    make_predictions_with_champions()