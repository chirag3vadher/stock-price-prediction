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
    print("---------test.py executed successfully-----------")
