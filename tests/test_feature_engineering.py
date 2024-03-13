# tests/test_feature_engineering.py

import unittest
from src.feature_engineering import calculate_technical_indicators

class TestFeatureEngineering(unittest.TestCase):
    def test_calculate_technical_indicators(self):
        # Test calculation of technical indicators
        # Assuming 'df' is a DataFrame with historical stock price data
        # Call calculate_technical_indicators function and assert the output
        df_with_indicators = calculate_technical_indicators(df)
        self.assertIsNotNone(df_with_indicators)
