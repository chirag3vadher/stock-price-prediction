"""
Stock price prediction: test.py

Dependencies:
    - logging: helps to create logs for execution
    - model_evaluation: module containing functions for evaluating machine learning models

Usage:
    1. Ensure that you have the required dependencies installed.
    2. Execute this module to evaluate the performance of trained machine learning models using test data.

Author Information:
    Name: Chirag Vadher
    Date: 12-03-2024
    Contact:

Abstract/Description:
    This module contains code to evaluate the performance of trained machine learning models using test data and metrics like MSE or MAPE.

Change Log:
    - 12-03-2024: Initial creation.
    - [Date]: Updated with new data or changes. -- Use this when updated
"""


from src import model_evaluation
from src.utils import load_model, logger
import pandas as pd
import numpy as np

def test_model(model_filename, X_test, y_test):
    """
  This function loads a saved model and evaluates its performance on test data.

  Args:
      model_filename (str): Filename of the saved model
      X_test (array): Test data features
      y_test (array): Test data labels
  """
    try:
        # Load the saved model
        loaded_model = load_model(model_filename)

        # Perform model evaluation (using functions from model_evaluation or model_utils)
        eval = model_evaluation.evaluate_model(loaded_model, X_test, y_test)

        #logging
        logger.info(f"test run is executed sucessfully, mse for the test is {eval}")

        return eval
    except Exception as e:
        logger.error(f"Error testing the model : {e}")
        return None


if __name__ == "__main__":
    # Example usage (assuming X_test, y_test and model filename are available)
    model_filename = "model.pkl"
    X_test = pd.DataFrame(np.load("X_test.npy"))
    y_test = pd.Series(np.load("y_test.npy"))
    test_model(model_filename, X_test, y_test)
