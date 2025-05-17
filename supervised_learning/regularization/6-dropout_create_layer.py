#!/usr/bin/env python3
"""
Create a Layer with Dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a dense layer with dropout.
    """
    init_weights = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_avg")

    layer = tf.keras.layers.Dense(
            units=n,
            activation=activation,
            kernel_initializer=init_weights
            )

    output = layer(prev)

    if training:
        dropout = tf.keras.layers.Dropout(rate=(1 - keep_prob))
        output = dropout(output, training=training)

    return output
