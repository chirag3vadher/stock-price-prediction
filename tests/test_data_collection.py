"""
Stock price prediction: test_data_collection.py

Dependencies:
    - unittest: framework for organizing and running unit tests
    - data_collection: module containing functions for fetching historical stock price data
    - logging: helps to create logs for execution

Usage:
    1. Ensure that you have the required dependencies installed.
    2. Import this module into your code to test the functionality of data collection functions.

Author Information:
    Name: Chirag Vadher
    Date: 12-03-2024
    Contact:

Abstract/Description:
    This module contains unit tests for the data collection module, ensuring that the functions for fetching historical stock price data work correctly.

Change Log:
    - 12-03-2024: Initial creation.
    - [Date]: Updated with new data or changes. -- Use this when updated
"""

import unittest
from src.data_collection import fetch_stock_data

class TestDataCollection(unittest.TestCase):
    def test_fetch_stock_data(self):
        # Test data fetching function
        data = fetch_stock_data("AAPL", "2022-01-01", "2022-12-31")
        self.assertIsNotNone(data)
