# tests/test_data_collection.py

import unittest
from src.data_collection import fetch_stock_data

class TestDataCollection(unittest.TestCase):
    def test_fetch_stock_data(self):
        # Test data fetching function
        data = fetch_stock_data("AAPL", "2022-01-01", "2022-12-31")
        self.assertIsNotNone(data)
