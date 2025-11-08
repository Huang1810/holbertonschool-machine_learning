#!/usr/bin/env python3
"""
Preprocess BTC data for RNN forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

def preprocess_data(filepaths, sequence_length=24):
    """
    Preprocess BTC datasets to create sequences for RNN forecasting.

    Args:
        filepaths (list): List of CSV file paths.
        sequence_length (int): Number of hours to use as input.

    Returns:
        X (np.ndarray): Input sequences of shape (num_samples, sequence_length, num_features)
        y (np.ndarray): Target values of shape (num_samples,)
        scaler (sklearn scaler): Scaler fitted on the data
    """
    # Load datasets and concatenate
    dfs = [pd.read_csv(fp) for fp in filepaths]
    df = pd.concat(dfs).sort_values('Timestamp').reset_index(drop=True)

    # Select useful columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']]

    # Rescale features to [0, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X = []
    y = []

    # Each sample: past `sequence_length` hours to predict next close price
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i, 3])  # Close price as target

    X = np.array(X)
    y = np.array(y)

    # Save preprocessed data
    np.save('X.npy', X)
    np.save('y.npy', y)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Saved preprocessed data: X.shape={X.shape}, y.shape={y.shape}")
    return X, y, scaler

if __name__ == '__main__':
    import sys
    filepaths = sys.argv[1:]  # Pass CSV file paths as command-line arguments
    preprocess_data(filepaths)
