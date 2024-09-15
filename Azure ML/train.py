import os
import argparse
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from azureml.core import Run

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(data_path):
    """
    Main function to execute the training pipeline.

    Args:
        data_path (str): Path to the CSV file containing the training data.

    Raises:
        ValueError: If there is an issue with data processing or model training.
    """
    logger.info('Starting training process.')

    # Get the experiment run context
    run = Run.get_context()

    try:
        # Load data from the provided path
        df = pd.read_csv(data_path)
        logger.info('Data loaded successfully.')

        # Data preprocessing
        # Drop rows where 'wh_est_year' or 'approved_wh_govt_certificate' are missing
        df.dropna(subset=['wh_est_year', 'approved_wh_govt_certificate'], inplace=True)
        logger.info('Dropped rows with missing target values.')

        # Impute missing values in 'workers_num' with the mean value
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        df.loc[:, ['workers_num']] = imputer.fit_transform(df.loc[:, ['workers_num']])
        logger.info('Missing values in workers_num imputed.')

        # Apply one-hot encoding to specified columns
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [0, 1, 2, 3, 8, 18])],
            remainder='passthrough'
        )
        transformed = ct.fit_transform(df)
        transformed_columns = ct.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=transformed_columns, index=df.index)
        logger.info('One-hot encoding applied.')

        # Prepare features (X) and target (y)
        X = transformed_df.iloc[:, :-1]  # All columns except the last one
        y = df.iloc[:, -1]  # Last column as the target variable

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        logger.info('Data split into training and testing sets.')

        # Train the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=10, random_state=0)
        rf_model.fit(X_train, y_train)
        logger.info('Random Forest model trained.')

        # Make predictions and calculate the Mean Absolute Percentage Error (MAPE)
        y_pred = rf_model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        logger.info(f'MAPE calculated: {mape:.4f}')

        # Log the MAPE metric to AzureML
        run.log('MAPE', mape)

        # Save the trained model to the 'outputs' directory
        os.makedirs('outputs', exist_ok=True)
        joblib.dump(rf_model, 'outputs/rf_model.joblib')
        logger.info('Model saved successfully.')

    except Exception as e:
        # Log the error and raise an exception
        logger.error(f'Error occurred: {e}', exc_info=True)
        raise

    finally:
        # Complete the AzureML run
        run.complete()
        logger.info('Run completed.')

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Train a Random Forest model on the provided dataset.')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the CSV file containing the training data')
    args = parser.parse_args()

    # Ensure the data path is provided
    if not args.data_path:
        logger.error('Data path is required.')
        raise ValueError('Data path is required.')

    # Execute the main function with the provided data path
    main(args.data_path)
