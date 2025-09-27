#!/usr/bin/env python3
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        # GRU layer (with glorot_uniform initializer)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self):
        """Initializes hidden state as zeros"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        x: tensor of shape (batch, input_seq_len)
        initial: tensor of shape (batch, units)
        Returns:
            outputs: (batch, input_seq_len, units)
            hidden: (batch, units)
        """
        x = self.embedding(x)  # (batch, input_seq_len, embedding_dim)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
