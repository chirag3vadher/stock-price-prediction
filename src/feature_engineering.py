"""
Stock price prediction: feature_engineering.py

Dependencies:
    - ta : Technical Analysis library useful to do feature engineering from financial time series datasets (Open, Close, High, Low, Volume)
    - pandas: library for data manipulation and analysis
    - numpy: library for numerical computations
    - logging: helps to create logs for execution

Usage:
    1. Ensure that you have the required dependencies installed.
    2. Import this module into your code to calculate financial indicators from historical stock price data.

Author Information:
    Name: Chirag Vadher
    Date: 12-03-2024
    Contact:

Abstract/Description:
    This module contains functions for calculating financial indicators such as moving averages and RSI from historical stock price data.

Change Log:
    - 12-03-2024: Initial creation.
    - [Date]: Updated with new data or changes. -- Use this when updated
"""


import pandas as pd
import os
from ta import add_all_ta_features
from src.utils import logger, PWD


def calculate_technical_indicators(df):
    """
    Calculate technical indicators from historical stock price data.

    Args:
        df (pd.DataFrame): DataFrame containing historical stock price data.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    try:
        # Calculate technical indicators using 'ta' library
        df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        logger.info("Technical indicators calculated successfully")
        return df
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return None


def save_processed_data(data, filename):
    """
    Save DataFrame to a CSV file.

    Args:
        data (pd.DataFrame): DataFrame to be saved.
        filename (str): Name of the file to save the DataFrame.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(f"{PWD}/../data/processed", exist_ok=True)

        # Save data to CSV file
        filepath = os.path.join(f"{PWD}/../data/processed", filename)
        data.to_csv(filepath, index=False)

        logger.info(f"Data saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")
