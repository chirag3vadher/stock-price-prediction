"""
Stock price prediction : src.utils.py

Dependencies:
    - Joblib: provides utilities for saving and loading Python objects
    - logging: helps to create the logs for execution
    - datetime - to handle date time
    - os - to execute os commands

Usage:
    1. Ensure that you have the required dependencies installed.

    Author Information:
    Name: Chirag Vadher
    Date: 12-03-2024
    Contact:

Abstract/Description:

    This code provides the utility functions for loading file, saving file and making prediction on data

Change Log:
    - 12 feb 2024: Initial creation.
    - [Date]: Updated with new data. -- Use this when updated

"""

import joblib
import logging
import datetime
import os

# Configure logging
PWD = os.path.dirname(__file__)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{PWD}/../data/logger/logs_{TIMESTAMP}.log"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')

# Get the logger object
logger = logging.getLogger(__name__)


def save_model(model, filename):
    """
    Save trained model to a file.

    Args:
        model: Trained machine learning model.
        filename (str): Name of the file to save the model.
    """
    try:
        joblib.dump(model, filename)
        logger.info(f"Model saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving model to {filename}: {e}")


def load_model(filename):
    """
    Load trained model from a file.

    Args:
        filename (str): Name of the file containing the saved model.

    Returns:
        model: Loaded machine learning model.
    """
    try:
        model = joblib.load(filename)
        logger.info(f"Model loaded from {filename}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filename}: {e}")
        return None


def make_prediction(model, X_new):
    """
    Make predictions using the deployed model.

    Args:
        model: Trained machine learning model.
        X_new (pd.DataFrame): Features of the new data.

    Returns:
        np.array: Predicted values.
    """
    try:
        predictions = model.predict(X_new)
        logger.info("Predictions made successfully")
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None
