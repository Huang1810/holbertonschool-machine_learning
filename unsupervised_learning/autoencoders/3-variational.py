#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) implementation
"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def sampling(args):
    """
    Reparameterization trick: sample from N(mu, sigma^2) using
    epsilon ~ N(0, I) and shift by learned parameters.
    """
    mu, log_var = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return mu + K.exp(0.5 * log_var) * epsilon


def build_encoder(input_dims, hidden_layers, latent_dims):
    """
    Build the encoder network.
    Returns a model outputting z, mu, log_var.
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims)(x)
    log_var = keras.layers.Dense(latent_dims)(x)

    z = keras.layers.Lambda(sampling)([mu, log_var])
    return keras.Model(inputs, [z, mu, log_var], name='encoder')


def build_decoder(latent_dims, hidden_layers, output_dims):
    """
    Build the decoder network.
    """
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(output_dims, activation='sigmoid')(x)
    return keras.Model(latent_inputs, outputs, name='decoder')


def vae_loss(inputs, outputs, mu, log_var, input_dims):
    """
    VAE loss: reconstruction loss + KL divergence.
    """
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + log_var - K.square(mu) - K.exp(log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
    return K.mean(reconstruction_loss + kl_loss)


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a VAE with encoder and decoder.
    """
    encoder = build_encoder(input_dims, hidden_layers, latent_dims)
    decoder = build_decoder(latent_dims, hidden_layers, input_dims)

    inputs = keras.Input(shape=(input_dims,))
    z, mu, log_var = encoder(inputs)
    outputs = decoder(z)

    auto = keras.Model(inputs, outputs, name='vae')
    auto.add_loss(vae_loss(inputs, outputs, mu, log_var, input_dims))
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
