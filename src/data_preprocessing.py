import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import logger


def preprocess_data(df):
    """
    Preprocess historical stock price data.

    Args:
        df (pd.DataFrame): DataFrame containing historical stock price data.

    Returns:
        pd.DataFrame: Preprocessed features for training.
        pd.DataFrame: Preprocessed features for testing.
        pd.Series: Target variable for training.
        pd.Series: Target variable for testing.
    """
    try:
        # Drop rows with missing values
        df.dropna(inplace=True)

        # Separate features and target variable
        X = df.drop(columns=['Close'])  # Assuming 'Close' is the target variable
        y = df['Close']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        logger.info("Data preprocessed successfully")
        return X_train_scaled_df, y_train, X_test_scaled_df, y_test
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None, None, None, None
