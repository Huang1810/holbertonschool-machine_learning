#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, models

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder.
    
    Args:
        input_dims (int): Dimensionality of the input.
        hidden_layers (list): List of integers for encoder hidden layer sizes.
        latent_dims (int): Dimensionality of the latent space.
        
    Returns:
        encoder (tf.keras.Model): The encoder model.
        decoder (tf.keras.Model): The decoder model.
        auto (tf.keras.Model): The full autoencoder model.
    """
    # Encoder
    input_layer = layers.Input(shape=(input_dims,))
    x = input_layer
    for nodes in hidden_layers:
        x = layers.Dense(nodes, activation='relu')(x)
    latent = layers.Dense(latent_dims, activation='relu')(x)
    
    encoder = models.Model(inputs=input_layer, outputs=latent, name="encoder")
    
    # Decoder
    decoder_input = layers.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = layers.Dense(nodes, activation='relu')(x)
    output_layer = layers.Dense(input_dims, activation='sigmoid')(x)
    
    decoder = models.Model(inputs=decoder_input, outputs=output_layer, name="decoder")
    
    # Autoencoder
    auto_input = input_layer
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = models.Model(inputs=auto_input, outputs=decoded, name="autoencoder")
    
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
