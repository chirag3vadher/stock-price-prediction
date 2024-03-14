"""
Stock price prediction: test_feature_engineering.py

Dependencies:
    - unittest: framework for organizing and running unit tests
    - feature_engineering: module containing functions for calculating financial indicators
    - logging: helps to create logs for execution

Usage:
    1. Ensure that you have the required dependencies installed.
    2. Import this module into your code to test the functionality of feature engineering functions.

Author Information:
    Name: Chirag Vadher
    Date: 12-03-2024
    Contact:

Abstract/Description:
    This module contains unit tests for the feature engineering module, ensuring that the functions for calculating financial indicators work correctly.

Change Log:
    - 12-03-2024: Initial creation.
    - [Date]: Updated with new data or changes. -- Use this when updated
"""

import unittest
from src.feature_engineering import calculate_technical_indicators
from src.data_collection import fetch_stock_data


class TestFeatureEngineering(unittest.TestCase):
    def test_calculate_technical_indicators(self):
        # Test calculation of technical indicators
        # Assuming 'df' is a DataFrame with historical stock price data
        # Call calculate_technical_indicators function and assert the output
        data = fetch_stock_data("AAPL", "2022-01-01", "2022-06-01")
        df_with_indicators = calculate_technical_indicators(data)
        self.assertIsNotNone(df_with_indicators)
