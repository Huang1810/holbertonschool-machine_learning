#!/usr/bin/env python3
""" Vanilla autoencoder """
import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates a vanilla autoencoder

        - input_dims is an integer containing the dimensions of
          the model input
        - hidden_layers is a list containing the number of nodes
          for each hidden layer in the encoder, respectively
        - the hidden layers should be reversed for the decoder
        - latent_dims is an integer containing the dimensions of
          the latent space representation
        Returns: encoder, decoder, auto
            - encoder is the encoder model
            - decoder is the decoder model
            - auto is the full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='linear')(x)

    # Decoder
    dec_input = keras.Input(shape=(latent_dims,))
    x = dec_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    # Encoder and decoder models
    encoder = keras.models.Model(inputs, latent)
    decoder = keras.models.Model(dec_input, outputs)

    # Autoencoder
    auto_input = inputs
    auto_output = decoder(encoder(auto_input))
    auto = keras.models.Model(auto_input, auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
