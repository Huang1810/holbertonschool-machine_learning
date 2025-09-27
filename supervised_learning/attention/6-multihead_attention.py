#!/usr/bin/env python3
"""
Multi Head Attention layer
"""
import tensorflow as tf
from tensorflow.keras import layers

# Import scaled dot product attention
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(layers.Layer):
    """Multi Head Attention class"""

    def __init__(self, dm, h):
        """
        Class constructor
        Args:
            dm: int representing the dimensionality of the model
            h: int representing the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        if dm % h != 0:
            raise ValueError("dm must be divisible by h")

        self.dm = dm
        self.h = h
        self.depth = dm // h

        # Dense layers to generate query, key, and value matrices
        self.Wq = layers.Dense(dm)
        self.Wk = layers.Dense(dm)
        self.Wv = layers.Dense(dm)

        # Final linear layer
        self.linear = layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth)
        and transpose the result to shape
        (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Forward pass for multi-head attention
        Args:
            Q: tensor of shape (batch, seq_len_q, dk)
            K: tensor of shape (batch, seq_len_v, dk)
            V: tensor of shape (batch, seq_len_v, dv)
            mask: always None
        Returns:
            output, weights
        """
        batch_size = tf.shape(Q)[0]

        # Linear projections
        Q = self.Wq(Q)  # (batch, seq_len_q, dm)
        K = self.Wk(K)  # (batch, seq_len_v, dm)
        V = self.Wv(V)  # (batch, seq_len_v, dm)

        # Split into heads
        Q = self.split_heads(Q, batch_size)  # (batch, h, seq_len_q, depth)
        K = self.split_heads(K, batch_size)  # (batch, h, seq_len_v, depth)
        V = self.split_heads(V, batch_size)  # (batch, h, seq_len_v, depth)

        # Apply scaled dot-product attention
        scaled_attention, weights = sdp_attention(Q, K, V, mask)
        # scaled_attention: (batch, h, seq_len_q, depth)

        # Transpose back and concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dm)
        )  # (batch, seq_len_q, dm)

        # Final linear layer
        output = self.linear(concat_attention)

        return output, weights
