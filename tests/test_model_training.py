# tests/test_model_building.py

import unittest
from src.model_building import train_linear_regression, train_random_forest

class TestModelBuilding(unittest.TestCase):
    def test_train_linear_regression(self):
        # Test Linear Regression model training
        # Assuming 'X_train' and 'y_train' are features and target variable respectively
        # Call train_linear_regression function and assert the output
        model = train_linear_regression(X_train, y_train)
        self.assertIsNotNone(model)

    def test_train_random_forest(self):
        # Test Random Forest model training
        # Assuming 'X_train' and 'y_train' are features and target variable respectively
        # Call train_random_forest function and assert the output
        model = train_random_forest(X_train, y_train)
        self.assertIsNotNone(model)
