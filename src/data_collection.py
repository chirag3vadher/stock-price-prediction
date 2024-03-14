"""
Stock price prediction: data_collection.py

Dependencies:
    - yfinance: provides utilities for fetching historical stock price data
    - logging: helps to create logs for execution
    - pandas: library for data manipulation and analysis
    - datetime: module for handling date and time objects

Usage:
    1. Ensure that you have the required dependencies installed.
    2. Import this module into your code to fetch historical stock price data.

Author Information:
    Name: Chirag Vadher
    Date: 12-03-2024
    Contact:

Abstract/Description:
    This module contains functions for fetching historical stock price data using the yfinance library.

Change Log:
    - 12-03-2024: Initial creation.
    - [Date]: Updated with new data or changes. -- Use this when updated
"""


import yfinance as yf
import os
from src.utils import logger, PWD

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch historical stock price data using yfinance.

    Args:
        symbol (str): Stock symbol (e.g., AAPL for Apple Inc.).
        start_date (str): Start date for fetching historical data (format: YYYY-MM-DD).
        end_date (str): End date for fetching historical data (format: YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame containing historical stock price data.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        logger.info(f"Stock data fetched successfully for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def save_raw_data(data, filename):
    """
    Save DataFrame to a CSV file.

    Args:
        data (pd.DataFrame): DataFrame to be saved.
        filename (str): Name of the file to save the DataFrame.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(f"{PWD}/../data/raw", exist_ok=True)

        # Save data to CSV file
        filepath = os.path.join(f"{PWD}/../data/raw", filename)
        data.to_csv(filepath, index=False)

        logger.info(f"Data saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")