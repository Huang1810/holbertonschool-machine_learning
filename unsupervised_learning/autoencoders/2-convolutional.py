#!/usr/bin/env python3
""" Convolutional autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ creates a convolutional autoencoder

        - input_dims is a tuple of integers containing the dimensions of the model input
        - filters is a list containing the number of filters for each conv layer in the encoder
        - filters should be reversed for the decoder
        - latent_dims is a tuple of integers containing the dimensions of the latent space
        Returns: encoder, decoder, auto
            - encoder is the encoder model
            - decoder is the decoder model
            - auto is the full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3,3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
    
    latent = keras.layers.Conv2D(latent_dims[2], (3,3), activation='relu', padding='same')(x)

    encoder = keras.models.Model(inputs, latent)

    # Decoder
    dec_input = keras.Input(shape=latent_dims)
    x = dec_input
    for i, f in enumerate(reversed(filters)):
        if i < len(filters) - 1:
            x = keras.layers.Conv2D(f, (3,3), activation='relu', padding='same')(x)
            x = keras.layers.UpSampling2D((2,2))(x)
        else:
            x = keras.layers.Conv2D(f, (3,3), activation='relu', padding='valid')(x)

    outputs = keras.layers.Conv2D(input_dims[2], (3,3), activation='sigmoid', padding='same')(x)

    decoder = keras.models.Model(dec_input, outputs)

    # Autoencoder
    auto_input = inputs
    auto_output = decoder(encoder(auto_input))
    auto = keras.models.Model(auto_input, auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
