#!/usr/bin/env python3
""" Convolutional auto-encoder """
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
    input_img = keras.Input(shape=input_dims)

    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu',
                            padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    for i in range(1, len(filters)):
        x = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    comp = x

    dec_input = keras.Input(shape=latent_dims)

    x = keras.layers.Conv2D(filters[-1], (3, 3), activation='relu',
                            padding='same')(dec_input)
    x = keras.layers.UpSampling2D((2, 2))(x)
    for i in range(len(filters) - 2, 0, -1):
        x = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu',
                            padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                                  padding='same')(x)

    encoder = keras.models.Model(input_img, comp)
    decoder = keras.models.Model(dec_input, decoded)

    # Fill the complete model and create it
    inp = encoder(input_img)
    outputs = decoder(inp)
    auto = keras.models.Model(input_img, outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
