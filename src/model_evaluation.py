# src/model_evaluation.py

from sklearn.metrics import mean_squared_error
from src.utils import logger


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model on validation data.

    Args:
        model: Trained machine learning model.
        X_val (pd.DataFrame): Features of the validation data.
        y_val (pd.Series): Target variable of the validation data.

    Returns:
        float: Mean Squared Error of the model predictions.
    """
    try:
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        logger.info("Model evaluated successfully")
        logger.info(f"model MSE is {mse}")
        return mse
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None
