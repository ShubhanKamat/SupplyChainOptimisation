from flask import Flask, request, jsonify
import pickle
import pandas as pd
import logging
import os
from transformation import preprocess_input

# Create a logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging to write logs to a file inside the 'logs' folder
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),  # Save logs to 'logs/app.log'
        logging.StreamHandler()  # Also output logs to the console
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model from the pickle file
model_path = 'models/random_forest_regressor.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Root route for checking if the API is running
@app.route('/')
def index():
    """
    Root endpoint to confirm that the API is up and running.
    Returns:
        str: A welcome message indicating that the API is live.
    """
    return "Welcome to the Supply Chain Prediction API!"

# Prediction route to handle POST requests for model inference
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions using the pre-trained model.
    
    Expects a JSON input with data that will be processed and passed to the model for predictions.

    Returns:
        json: A JSON response with the model's prediction or an error message if something goes wrong.
    """
    try:
        # Get the input data from the request in JSON format
        input_data = request.json

        # Convert the input JSON into a Pandas DataFrame
        df = pd.DataFrame(input_data)

        # Preprocess the input data using the preprocessing pipeline defined in transformations.py
        processed_data = preprocess_input(df)

        # Use the loaded model to make predictions on the preprocessed data
        prediction = model.predict(processed_data)

        # Log the prediction
        logger.info(f"Prediction made: {prediction.tolist()}")

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        # Log the error message for debugging and return the error message to the client
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

# Main entry point for running the Flask app
if __name__ == '__main__':
    """
    When this script is executed, it will start the Flask server.
    Flask runs in debug mode and listens on all interfaces (0.0.0.0), making it accessible externally.
    """
    app.run(debug=True, host='0.0.0.0')
