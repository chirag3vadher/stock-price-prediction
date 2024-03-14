"""
Stock price prediction: test_model_evaluation.py

Dependencies:
    - unittest: framework for organizing and running unit tests
    - model_evaluation: module containing functions for evaluating machine learning models
    - logging: helps to create logs for execution

Usage:
    1. Ensure that you have the required dependencies installed.
    2. Import this module into your code to test the functionality of model evaluation functions.

Author Information:
    Name: Chirag Vadher
    Date: 12-03-2024
    Contact:

Abstract/Description:
    This module contains unit tests for the model evaluation module, ensuring that the functions for evaluating machine learning models work correctly.

Change Log:
    - 12-03-2024: Initial creation.
    - [Date]: Updated with new data or changes. -- Use this when updated
"""


import unittest
from src.model_evaluation import evaluate_model
from src.data_collection import fetch_stock_data
from src.feature_engineering import calculate_technical_indicators
from src.data_preprocessing import preprocess_data
from src.model_building import train_random_forest

class TestModelEvaluation(unittest.TestCase):
    def test_evaluate_model(self):
        # Test model evaluation function
        # Call evaluate_model function and assert the output
        data = fetch_stock_data("AAPL", "2022-01-01", "2022-06-02")
        df = calculate_technical_indicators(data)
        X_train, y_train, X_val, y_val = preprocess_data(df)
        model = train_random_forest(X_train, y_train)
        mse = evaluate_model(model, X_val, y_val)
        self.assertIsInstance(mse, float)
