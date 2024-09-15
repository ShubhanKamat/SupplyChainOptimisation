# FMCG Supply chain Analysis using Machine Learning and Azure ML

This repository contains a production-ready Flask app, as well as a detailed model training notebook for serving a Random Forest Regressor model trained on comprehensive information related to the instant noodles business of a leading FMCG company.

## Configuration

This project requires two configuration files:

1. `config.json`: Contains sensitive information.

2. `environment.yml`: Specifies the Python environment for this project.

**Note:** For security reasons, `config.json` and `environment.yml` are not included in this repository.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model APIs](#model-apis)
- [Transformation Pipeline](#transformation-pipeline)
- [Deployment](#deployment)
- [License](#license)

## Project Overview

This project is designed to serve a a Random Forest Regressor model trained on comprehensive information related to the instant noodles business of a leading FMCG company
 via a Flask API, as well as provides scripts for training the model using the azure python sdk. A detailed data exploration and model training notebook is available in the 'notebook' folder. The raw data is available under the 'data' folder. The data was sourced from kaggle. 
 
In addition to serving the model, the repository includes:
- **Transformation Pipeline:** To preprocess the data for models.


## Folder Structure

```bash
project-root/
│
├── app/                      # Flask app
│   ├── __init__.py            # Initializes the app
│   ├── app.py                 # Flask API for serving models
│   ├── transformation.py      # Data preprocessing steps
│
├── Azure ML/                      # Directory with the files for Azure python SDK code
│   ├── data/                      # Dataset directory
|        ├── FMCG_data.csv         # Dataset
|   ├── outputs/                   #joblib file with the model
│   ├── run_experiment.py          # Script to run the experiment on Azure ML
│   ├── train.py                   # Training script
|
├── data/                      # Data storage 
│   ├── FMCG_data.csv                  # Raw data file        
│
├── models/                    # Model training scripts
│   ├── random_forest_regressor.pkl         # Trained ML model
|   ├── imputer.pkl                         #Imputing data
|   ├── transformer.pkl                     # for preprocessing
│
├── scripts/                   # Helper scripts for running app & pipelines
│   ├── run_app.sh             # Start the Flask app
│
├── notebook/                 # Jupyter notebooks for EDA & experiments
│   └── Model_trainer.ipynb
│
├── requirements.txt           # Python package requirements
├── README.md                  # Project documentation
└── wsgi.py                    # WSGI entry point for production deployment
```
## Features

- **Model Serving:** Serve a trained Random Forest Regressor via a Flask API.
- **Data Transformation:** Preprocess data to prepare it for model inference.
- **Modular Architecture:** Clear separation of concerns for transformation, and model serving.
- **Scalable:** Ready for deployment in production environments (WSGI, Docker, etc.).

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/ShubhanKamat/SupplyChainOptimisation.git
    cd SupplyChainOptimisation
    ```

2. **Set Up Python Virtual Environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```


## Usage

1. **Running the Flask App**

    To start the Flask app locally, run:

    ```bash
    bash scripts/run_app.sh
    ```

    Alternatively, you can run:

    ```bash
    export FLASK_APP=app/app.py
    export FLASK_ENV=production  # Set environment for production
    flask run
    ```

    By default, the Flask API will be served at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Model APIs

The Flask app exposes a route for interacting with the model. The model is preloaded into memory when the app starts.

**Endpoint:**

- `POST /predict`: Predicts using the regressor model post preprocessing.


**Example Request:**

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
    "data": [your_data]
}'
```
## Response

```json
{
  "prediction": [ predicted_value ]
}
```

## Transformation Pipeline
The transformation pipeline preprocesses raw data for model consumption.

To manually run data transformation:

```bash
python app/transformation.py
```

## Deployment
This app is ready for production deployment using WSGI servers like Gunicorn or uWSGI.

Gunicorn Deployment Example:

```bash
gunicorn --bind 0.0.0.0:8000 wsgi:app
```
The app will be accessible at http://localhost:8000 on your local machine.

You can test the endpoints using curl or Postman.


## License
This project is licensed under the MIT License - see the LICENSE file for details.
