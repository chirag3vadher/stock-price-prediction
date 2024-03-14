"""
Stock price prediction: test_data_preprocessing.py

Dependencies:
    - unittest: framework for organizing and running unit tests
    - data_preprocessing: module containing functions for preprocessing data
    - logging: helps to create logs for execution

Usage:
    1. Ensure that you have the required dependencies installed.
    2. Import this module into your code to test the functionality of data preprocessing functions.

Author Information:
    Name: Chirag Vadher
    Date: 12-03-2024
    Contact:

Abstract/Description:
    This module contains unit tests for the data preprocessing module, ensuring that the functions for cleaning, handling missing values, and normalizing data work correctly.

Change Log:
    - 12-03-2024: Initial creation.
    - [Date]: Updated with new data or changes. -- Use this when updated
"""


import unittest
from src.data_collection import fetch_stock_data
from src.feature_engineering import calculate_technical_indicators
from src.data_preprocessing import preprocess_data


class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        data = fetch_stock_data("AAPL", "2022-01-01", "2022-02-20")
        df = calculate_technical_indicators(data)
        X_train, y_train, X_val, y_val = preprocess_data(df)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_val)
