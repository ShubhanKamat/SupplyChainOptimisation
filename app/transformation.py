import pandas as pd
import pickle
import logging
import os

# Create a logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging to write logs to a file inside the 'logs' folder
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/transformation.log'),  # Save logs to 'logs/app.log'
        logging.StreamHandler()  # Also output logs to the console
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)

# Load the pre-trained imputer and transformer from pickle files
imputer_path = 'models/imputer.pkl'
transformer_path = 'models/transformer.pkl'

with open(imputer_path, 'rb') as imputer_file:
    imputer = pickle.load(imputer_file)

with open(transformer_path, 'rb') as transformer_file:
    transformer = pickle.load(transformer_file)

def preprocess_input(df):
    """
    Preprocess the input DataFrame for inference using the pre-trained imputer and transformer.
    Args:
        df (pd.DataFrame): Input data for inference
    Returns:
        np.ndarray: Transformed data ready for model prediction
    """
    try:
        logger.info('Starting preprocessing of input data.')

        # Drop rows where 'wh_est_year' or 'approved_wh_govt_certificate' are missing
        df.dropna(subset=['wh_est_year', 'approved_wh_govt_certificate'], inplace=True)
        logger.info('Dropped rows with missing target values.')

        # Impute missing values in 'workers_num' using the pre-trained imputer
        df.loc[:, ['workers_num']] = imputer.transform(df.loc[:, ['workers_num']])
        logger.info('Missing values in workers_num imputed.')

        # Apply the pre-trained transformer (which includes OneHotEncoder)
        transformed = transformer.transform(df)
        logger.info('One-hot encoding applied.')

        return transformed

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise ValueError(f"Error during preprocessing: {e}")
