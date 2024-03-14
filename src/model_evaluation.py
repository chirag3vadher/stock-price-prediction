
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
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
        logger.info(f"{type(model).__name__} evaluated successfully")
        logger.info(f"{type(model).__name__} is {mse}")
        return mse
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None


def select_best_model(models, X_val, y_val):
    """
    Select the best performing model based on evaluation metrics.

    Args:
        models (list): List of trained machine learning models.
        X_val (pd.DataFrame): DataFrame containing features for validation.
        y_val (pd.Series): Series containing target variable for validation.

    Returns:
        object: Best performing trained model.
    """
    best_model = None
    best_mse = float('inf')

    try:
        for model in models:
            mse = evaluate_model(model, X_val, y_val)
            if mse < best_mse:
                best_mse = mse
                best_model = model
    except Exception as e:
        logger.error(f"Error selecting best model: {e}")
        best_model = None

    return best_model

def tune_hyperparameters(model, param_grid, X_train, y_train):
    """
    Perform hyperparameter tuning for the given model using Grid Search.

    Args:
        model: Machine learning model.
        param_grid (dict): Dictionary containing hyperparameter grid for Grid Search.
        X_train (pd.DataFrame): DataFrame containing features for training.
        y_train (pd.Series): Series containing target variable for training.

    Returns:
        GridSearchCV: Grid Search object containing tuned model.
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search