# src/model_building.py

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
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

def train_lstm(X_train, y_train):
    """
    Train an LSTM model on the training data.

    Args:
        X_train (np.ndarray): Array containing features for training (for LSTM).
        y_train (np.ndarray): Array containing target variable for training (for LSTM).

    Returns:
        Sequential: Trained LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    return model
