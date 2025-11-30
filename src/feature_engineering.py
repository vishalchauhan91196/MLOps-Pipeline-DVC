import os
import sys

# Python also looks for imports in the root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.logger import get_logger
from utils.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

logger = get_logger('feature_engineering')

def load_data(file_path: str) -> pd.DataFrame:
    """ Load data from a CSV file. """
    try:
        logger.debug(f'----- Starting to load data from {INTERIM_DATA_DIR} -----')

        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded and NaNs filled from %s", file_path)
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


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """ Apply TfIdf to the data. """
    try:
        logger.debug('Starting applying tfidf to data')

        vectorizer = TfidfVectorizer(max_features = max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['labels'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['labels'] = y_test

        logger.debug('tfidf applied and data transformed')
        return train_df, test_df

    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """ Save the train and test datasets. """
    try:
        train_data.to_csv(f"{PROCESSED_DATA_DIR}/train_tfidf.csv", index=False)
        test_data.to_csv(f"{PROCESSED_DATA_DIR}/test_tfidf.csv", index=False)
        logger.debug('Train and test data saved to %s', PROCESSED_DATA_DIR)

    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


def main():
    try:
        max_features = 40

        # Load interim data
        train_data = load_data(f'{INTERIM_DATA_DIR}/train_processed.csv')
        test_data  = load_data(f'{INTERIM_DATA_DIR}/test_processed.csv')

        # Apply TFIDF
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        # Save data inside data/raw
        save_data(train_df, test_df)
        
    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)
        print(f'Error: {e}')   

if __name__ == "__main__":
    main()