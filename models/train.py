from src import data_collection, feature_engineering, data_preprocessing, model_building, model_evaluation
from src.utils import save_model, logger
import numpy as np


def train_model(symbol, start_date, end_date):
    """
  This function trains a model using the provided data.

  Args:
      symbol (str): Stock symbol
      start_date (str): Start date in YYYY-MM-DD format
      end_date (str): End date in YYYY-MM-DD format

  Returns:
      object: Trained model object
  """
    try:
        # Fetch historical stock price data and save data
        data = data_collection.fetch_stock_data(symbol, start_date, end_date)
        data_collection.save_raw_data(data, f"raw_{symbol}_{start_date}_{end_date}")

        # Perform feature engineering
        features = feature_engineering.calculate_technical_indicators(data)
        feature_engineering.save_processed_data(features,f"processed_{symbol}_{start_date}_{end_date}")

        # Preprocess the data
        X_train, y_train, X_test, y_test = data_preprocessing.preprocess_data(features)

        # Train the model
        models = []
        models.append(model_building.train_linear_regression(X_train, y_train))
        models.append(model_building.train_random_forest(X_train, y_train))
        X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))  # Reshape for LSTM
        models.append(model_building.train_lstm(X_train_lstm, y_train))

        # validate_the_model
        model_evaluation.evaluate_model(model, X_test, y_test)

        # Save the trained model
        model_filename = "model.pkl"
        save_model(model, model_filename)
        print("Model saved successfully")

        return model
    except Exception as e:
        logger.error(f"Error tarining the model {symbol}: {e}")
        return None


if __name__ == "__main__":
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    trained_model = train_model(symbol, start_date, end_date)
