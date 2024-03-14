"""
Stock Price Prediction API

This Flask application serves as an API for predicting stock prices using a pre-trained machine learning model.
It exposes an endpoint '/predict' for making predictions based on input data provided in a POST request.

Dependencies:
    - Flask: Web framework for creating the API endpoints
    - src.utils: Module containing utility functions for loading the trained model and making predictions

Usage:
    - Ensure that the required dependencies are installed.
    - Start the Flask application by running this script.

Author: Chirag Vadher
Date: 12 Mar 2024

"""


import numpy as np
from flask import Flask, request, jsonify
from src.utils import load_model, make_prediction

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = load_model("model.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for making predictions.

    Receives input data from a POST request, preprocesses the data,
    makes predictions using a trained model, and returns predictions
    as a JSON response.

    Returns:
        JSON: Predictions as a JSON response.
    """
    try:
        # Get input data from request
        input_data = request.json

        # Preprocess the data
        preprocessed_data = np.array(input_data)

        # Make predictions
        predictions = make_prediction(model, preprocessed_data)

        # Return predictions as JSON response
        return jsonify(predictions=predictions)

    except Exception as e:
        # Return error message in case of any exception
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    # Run the Flask application
    app.run(host="0.0.0.0", port=8080, debug=True)

