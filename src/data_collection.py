import yfinance as yf
import os
from src.utils import logger, pwd

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
        os.makedirs(f"{pwd}/../data/raw", exist_ok=True)

        # Save data to CSV file
        filepath = os.path.join(f"{pwd}/../data/raw", filename)
        data.to_csv(filepath, index=False)

        logger.info(f"Data saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")