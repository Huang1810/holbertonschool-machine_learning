#!/usr/bin/env python3
"""
Train an RNN to forecast BTC close price.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle

# Load preprocessed data
X = np.load('X.npy')
y = np.load('y.npy')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create tf.data.Dataset
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# Build the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Predict single close price
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20)

# Save the trained model
model.save('btc_rnn_model.h5')
print("Model saved as btc_rnn_model.h5")
