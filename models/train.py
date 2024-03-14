from src import data_collection, feature_engineering, data_preprocessing, model_building, model_evaluation
from src.utils import save_model, logger


def train_model(symbol, start_date, end_date):
    """
  This function trains a model using the provided data does the finetuning and saves the best model.

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
        X_train, y_train, X_val, y_val = data_preprocessing.preprocess_data(features)

        # Train the model
        models = []
        models.append(model_building.train_linear_regression(X_train, y_train))
        models.append(model_building.train_random_forest(X_train, y_train))

        #validate the model
        best_model = model_evaluation.select_best_model(models, X_val, y_val)
        logger.info(f"Best model is {type(best_model).__name__}")
        tuned_model = None
        if best_model is not None:
            # Hyperparameter Tuning
            if type(best_model).__name__ == "RandomForestRegressor":
                param_grid = {'n_estimators': [100, 200, 300],
                              'max_depth': [None, 10, 20]}
                tuned_model = model_evaluation.tune_hyperparameters(best_model, param_grid, X_train, y_train)
                logger.info("Tuned Random Forest Model:", tuned_model.best_params_)
            elif type(best_model).__name__ == "LinearRegression":
                param_grid = {'fit_intercept': [True, False]}
                tuned_model = model_evaluation.tune_hyperparameters(best_model, param_grid, X_train,y_train)
                logger.info("Tuned Linear Regression Model:", tuned_model.best_params_)
            else:
                logger.info("Best model does not support hyperparameter tuning.")
        else:
            logger.info("No best model selected.")

        # Save the tuned model
        if tuned_model is not None:
            model_filename = "model.pkl"
            save_model(tuned_model, model_filename)
            logger.info("tuned model saved successfully")

    except Exception as e:
        logger.error(f"Error tarining the model {symbol}: {e}")
        return None


if __name__ == "__main__":
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    trained_model = train_model(symbol, start_date, end_date)
