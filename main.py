# main.py
from src import data_collection, feature_engineering, data_preprocessing, model_building, model_evaluation
from src.utils import save_model, load_model, make_prediction

def main():
    try:
        # Fetch historical stock price data
        symbol = "AAPL"  # Example stock symbol (Apple Inc.)
        start_date = "2022-01-01"
        end_date = "2022-12-31"
        data = data_collection.fetch_stock_data(symbol, start_date, end_date)

        # Perform feature engineering
        features = feature_engineering.calculate_technical_indicators(data)

        # Preprocess the data
        X_train, y_train, X_test, y_test = data_preprocessing.preprocess_data(features)

        # Train the model
        model = model_building.train_random_forest(X_train, y_train)

        # Save the trained model
        model_filename = "model.pkl"
        save_model(model, model_filename)
        print("Model saved successfully")

        # Load the saved model
        loaded_model = load_model(model_filename)

        # Make predictions using the loaded model
        predictions = make_prediction(loaded_model, X_test)
        print("Predictions:", predictions)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
