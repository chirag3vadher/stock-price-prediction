# tests/test_model_evaluation.py

import unittest
from src.model_evaluation import evaluate_model

class TestModelEvaluation(unittest.TestCase):
    def test_evaluate_model(self):
        # Test model evaluation function
        # Assuming 'model' is a trained machine learning model, and 'X_val' and 'y_val' are validation data
        # Call evaluate_model function and assert the output
        mse = evaluate_model(model, X_val, y_val)
        self.assertIsInstance(mse, float)
