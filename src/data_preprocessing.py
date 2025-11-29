import os
import sys

# Python also looks for imports in the root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import pandas as pd 
import string
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from utils.logger import get_logger
from utils.config import RAW_DATA_DIR, INTERIM_DATA_DIR

nltk.download('stopwords')
nltk.download("punkt")

logger = get_logger('data_preprocessing')

def transform_text(text):
    """ Transforms input text by converting it to lowercase, tokenizing, removing stopwords & punctuation , and stemming. """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords & punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join tokens back into a single string
    return " ".join(text)


def preprocess_df(df, name, text_column='text', target_column='target') -> pd.DataFrame:
    """ Preprocess the dataframe by encoding the target column, removing duplicates and transforming the text column. """
    try:
        logger.debug(f"----- Starting preprocessing for {name} -----")

        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Target Column encoded")

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicates removed")

        # Apply text transformation to specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug(f"----- Finished preprocessing for {name} -----")
        return df

    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise

    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise


def main(text_column='text', target_column='target'):
    """ Main function to load raw data, preprocess & transform it and save the processed data. """
    try:
        # Fetch data from data/raw
        train_data = pd.read_csv(f'{RAW_DATA_DIR}/train.csv')
        test_data = pd.read_csv(f'{RAW_DATA_DIR}/test.csv')
        logger.debug("Train and test data loaded properly from data/raw")

        # Transform the data
        train_processed_data = preprocess_df(train_data, "TRAIN", text_column, target_column)
        test_processed_data = preprocess_df(test_data, "TEST", text_column, target_column)
        logger.debug('Preprocessing of DataFrame completed')

        # Save data inside data/interim
        train_processed_data.to_csv(f"{INTERIM_DATA_DIR}/train_processed.csv", index=False)
        logger.debug('Train processed data saved to %s', INTERIM_DATA_DIR)
        test_processed_data.to_csv(f"{INTERIM_DATA_DIR}/test_processed.csv", index=False)
        logger.debug('Test processed data saved to %s', INTERIM_DATA_DIR)

    except FileNotFoundError as e:
        logger.error("File not found. %s", e)
        raise

    except pd.errors.EmptyDataError as e:
        logger.error("No data. %s", e)
        raise    

    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)
        print(f'Error: {e}')   

if __name__ == "__main__":
    main()    