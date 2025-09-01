import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras import layers, models

def run_isolation_forest(X: pd.DataFrame, contamination: float = 0.01) -> np.ndarray:
    """
    Fits an Isolation Forest model and returns anomaly scores.

    Args:
        X (pd.DataFrame): The input feature matrix.
        contamination (float): The proportion of outliers expected in the data.

    Returns:
        np.ndarray: Anomaly scores, where higher values indicate a higher likelihood of being an anomaly.
    """
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(X)
    # The decision_function values are inverted to make them more intuitive (higher = more anomalous)
    return -model.decision_function(X)

def run_local_outlier_factor(X: pd.DataFrame, contamination: float = 0.01) -> np.ndarray:
    """
    Fits a Local Outlier Factor (LOF) model and returns anomaly predictions.

    Args:
        X (pd.DataFrame): The input feature matrix.
        contamination (float): The proportion of outliers expected in the data.

    Returns:
        np.ndarray: An array where -1 indicates an anomaly and 1 indicates a normal point.
    """
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    return model.fit_predict(X)

def build_autoencoder(input_dim: int) -> models.Model:
    """
    Builds and compiles a simple Autoencoder model using Keras.

    Args:
        input_dim (int): The number of features in the input data.

    Returns:
        A compiled Keras Autoencoder model.
    """
    # Encoder
    inp = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(inp)
    encoded = layers.Dense(32, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = models.Model(inp, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder

def get_reconstruction_errors(model: models.Model, X: pd.DataFrame) -> np.ndarray:
    """
    Calculates the reconstruction error for each data point using a trained autoencoder.

    Args:
        model (models.Model): The trained Keras Autoencoder model.
        X (pd.DataFrame): The input feature matrix.

    Returns:
        np.ndarray: An array of mean squared errors (reconstruction errors).
    """
    reconstructed_X = model.predict(X)
    mse = np.mean(np.power(X.values - reconstructed_X, 2), axis=1)
    return mse