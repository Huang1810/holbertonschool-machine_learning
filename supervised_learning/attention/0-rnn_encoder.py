#!/usr/bin/env python3
"""
0-rnn_encoder.py
RNN Encoder module for supervised learning sequence models.
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNEncoder class for encoding sequences using a recurrent neural network.

    This encoder uses an embedding layer followed by a GRU layer to
    convert input sequences of token indices into context-rich hidden states.

    Attributes:
        vocab_size (int): Size of the input vocabulary.
        embedding_dim (int): Dimensionality of the embedding layer.
        enc_units (int): Number of units in the GRU layer.
        batch_sz (int): Batch size for training.
        embedding (tf.keras.layers.Embedding): Embedding layer.
        gru (tf.keras.layers.GRU): GRU layer for sequence encoding.
    """

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        """
        Initialize the RNNEncoder.

        Args:
            vocab_size (int): Size of the input vocabulary.
            embedding_dim (int): Dimensionality of the embedding layer.
            enc_units (int): Number of units in the GRU layer.
            batch_sz (int): Batch size for training.
        """
        super(RNNEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x, hidden):
        """
        Forward pass of the RNNEncoder.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_len).
            hidden (tf.Tensor): Initial hidden state of shape (batch_size, enc_units).

        Returns:
            output (tf.Tensor): All GRU outputs for each timestep, shape (batch_size, seq_len, enc_units).
            state (tf.Tensor): Final hidden state, shape (batch_size, enc_units).
        """
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        """
        Initialize the hidden state to zeros.

        Returns:
            tf.Tensor: Zero-initialized hidden state of shape (batch_sz, enc_units).
        """
        return tf.zeros((self.batch_sz, self.enc_units))
