# tests/test_data_preprocessing.py

import unittest
from src.data_preprocessing import preprocess_data
from src.utils import PWD
class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        # Test data preprocessing function
        # Assuming 'X' and 'y' are features and target variable respectively
        # Call preprocess_data function and assert the output
        X_processed, y_processed = preprocess_data(X, y)
        self.assertIsNotNone(X_processed)
        self.assertIsNotNone(y_processed)
