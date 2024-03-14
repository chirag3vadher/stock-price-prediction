"""
Stock price prediction: model_building.py

Dependencies:
    - sklearn: library for machine learning models
    - logging: helps to create logs for execution

Usage:
    1. Ensure that you have the required dependencies installed.
    2. Import this module into your code to train machine learning models.

Author Information:
    Name: Chirag Vadher
    Date: 12-03-2024
    Contact:

Abstract/Description:
    This module contains functions for training machine learning models such as Linear Regression, Random Forest, and LSTM.

Change Log:
    - 12-03-2024: Initial creation.
    - [Date]: Updated with new data or changes. -- Use this when updated
"""


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.utils import logger


def train_linear_regression(X_train, y_train):
    """
    Train Linear Regression model.

    Args:
        X_train (pd.DataFrame): Features of the training data.
        y_train (pd.Series): Target variable of the training data.

    Returns:
        sklearn.linear_model.LinearRegression: Trained Linear Regression model.
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.info("Linear Regression model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training Linear Regression model: {e}")
        return None


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.

    Args:
        X_train (pd.DataFrame): Features of the training data.
        y_train (pd.Series): Target variable of the training data.

    Returns:
        sklearn.ensemble.RandomForestRegressor: Trained Random Forest model.
    """
    try:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        logger.info("Random Forest model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training Random Forest model: {e}")
        return None
