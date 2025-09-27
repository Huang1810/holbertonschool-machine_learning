#!/usr/bin/env python3
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        # Dense layers for alignment model
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        s_prev: tensor of shape (batch, units) - previous decoder hidden state
        hidden_states: tensor of shape (batch, input_seq_len, units) - encoder outputs

        Returns:
            context: (batch, units) - context vector
            weights: (batch, input_seq_len, 1) - attention weights
        """
        # Expand s_prev to (batch, 1, units) for broadcasting
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Alignment score (energy)
        score = self.V(
            tf.nn.tanh(
                self.W(s_prev_expanded) + self.U(hidden_states)
            )
        )  # shape: (batch, input_seq_len, 1)

        # Attention weights
        weights = tf.nn.softmax(score, axis=1)  # normalize across input_seq_len

        # Context vector (weighted sum of encoder hidden states)
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # (batch, units)

        return context, weights
