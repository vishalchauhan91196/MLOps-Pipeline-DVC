import os
import sys

# Python also looks for imports in the root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import pandas as pd
import numpy as numpy
import pickle
from sklearn.ensemble import RandomForestClassifier

from utils.config import PROCESSED_DATA_DIR, MODEL_DIR
from utils.logger import get_logger

logger = get_logger('model_building')

def load_data(file_path: str) -> pd.DataFrame:
    """ Load data from a CSV file. """
    try:
        logger.debug(f'----- Starting to load data from {PROCESSED_DATA_DIR} -----')

        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s with shape %s", file_path, df.shape)
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


def train_model(X_train: np.ndarray, y_train: np.ndarray):
    """ Train the RandomForest model.
        X_train: Training features
        y_train: Training labels """

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")

        logger.debug('Initializing RandomForest model')
        clf = RandomForestClassifier(n_estimators=50, random_state=24)

        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')

        return clf

    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise

    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise    


def save_model(model, model_path: str) -> None:
    """ Save the trained model to a file. """
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
            
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise

    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def main():
    try:
        # Load processed train data
        train_data = load_data(f'{PROCESSED_DATA_DIR}/train_tfidf.csv')
        X_train = train_data.iloc[: , :-1].values
        y_train = train_data.iloc[: , -1].values

        # Train Model
        clf = train_model(X_train, y_train)

        # Save model
        model_path = f'{MODEL_DIR}/model.pkl'
        save_model(clf, model_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")    

if __name__ == '__main__':
    main()        