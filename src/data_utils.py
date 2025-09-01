import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def load_and_prepare_finance_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and prepares the financial transaction data for anomaly detection.

    Args:
        path (str): The file path to the financial transactions CSV.

    Returns:
        A tuple containing the original DataFrame and the scaled feature matrix (X).
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    
    # Select features and scale them
    features = df[["amount", "merchant_code", "account_age_days"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    return df, pd.DataFrame(X_scaled, columns=features.columns)

def load_and_prepare_iot_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and prepares the IoT sensor data for anomaly detection.

    Args:
        path (str): The file path to the IoT sensor readings CSV.

    Returns:
        A tuple containing the original DataFrame and the scaled feature matrix (X).
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    
    # Select features and scale them
    features = df[["sensor1", "sensor2", "temperature"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    return df, pd.DataFrame(X_scaled, columns=features.columns)