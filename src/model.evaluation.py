import os
import sys

# Python also looks for imports in the root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import pandas as pd
import numpy as numpy
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from utils.config import MODEL_DIR, PROCESSED_DATA_DIR, MODEL_METRICS_DIR
from utils.logger import get_logger

logger = get_logger('model_evaluation')

def load_model(model_path: str):
    """ Load trained model. """
    try:
        logger.debug(f'----- Starting to load model from {MODEL_DIR} -----')

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        logger.debug("Model loaded from %s", model_path)
        return model

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise   
 
    except Exception as e:
        logger.error("Unexpected error occurred while loading the model: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """ Load data from a CSV file. """
    try:
        logger.debug(f'----- Starting to load data from {PROCESSED_DATA_DIR} -----')

        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df

    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file. %s", e)
        raise

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise   
 
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        auc       = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        return metrics

    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise    


def save_metrics(metrics: dict, metrics_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file)

        logger.debug('Metrics saved to %s', metrics_path)    

    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def main():
    try:
        # Load processed test data
        test_data = load_data(f'{PROCESSED_DATA_DIR}/test_tfidf.csv')
        X_test = test_data.iloc[: , :-1].values
        y_test = test_data.iloc[: , -1].values

        # Load the trained model
        model_path = f'{MODEL_DIR}/model.pkl'
        clf = load_model(model_path)

        # Evaluate Model
        metrics = evaluate_model(clf, X_test, y_test)

        # Save metrics
        metrics_path = f'{MODEL_METRICS_DIR}/metrics.json'
        save_metrics(metrics, metrics_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")    

if __name__ == '__main__':
    main()            