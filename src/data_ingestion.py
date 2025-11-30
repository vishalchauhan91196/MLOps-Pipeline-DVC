import os
import sys

# Python also looks for imports in the root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from utils.logger import get_logger
from utils.config import RAW_DATA_DIR


logger = get_logger('data_ingestion')


def load_data(data_url: str) -> pd.DataFrame:
    """ Load data from a CSV file. """
    try:
        logger.debug('----- Starting Data Ingestion -----')

        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Pre-process the data. """
    try:
        df.drop(columns={'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'}, inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug("Data preprocessed successfully")
        return df

    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise       

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """ Save the train and test datasets. """
    try:
        train_data.to_csv(f"{RAW_DATA_DIR}/train.csv", index=False)
        test_data.to_csv(f"{RAW_DATA_DIR}/test.csv", index=False)
        logger.debug('Train and test data saved to %s', RAW_DATA_DIR)

    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise     

def main():
    """ Main function to load data, preprocess it and save the raw data. """
    try:
        test_size=0.2
        data_path = "https://raw.githubusercontent.com/vishalchauhan91196/MLOps-Pipeline-DVC/refs/heads/master/experiments/spam.csv"

        # Load data from source data path
        df = load_data(data_path)

        # Preprocess the data
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=24)

        # Save data inside data/raw
        save_data(train_data, test_data)
        
    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        print(f'Error: {e}')   

if __name__ == "__main__":
    main()