#!/usr/bin/env python3
"""
Transformer Encoder Block
"""
import tensorflow as tf
from tensorflow.keras import layers

# Import MultiHeadAttention
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(layers.Layer):
    """Encoder Block for Transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        Args:
            dm: int, model dimensionality
            h: int, number of attention heads
            hidden: int, number of hidden units in fully connected layer
            drop_rate: float, dropout rate
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = layers.Dense(hidden, activation="relu")
        self.dense_output = layers.Dense(dm)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass for encoder block
        Args:
            x: tensor of shape (batch, input_seq_len, dm)
            training: boolean, whether in training mode
            mask: optional mask
        Returns:
            tensor of shape (batch, input_seq_len, dm)
        """
        # Multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)  # self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # residual connection + norm

        # Feed-forward network
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)

        out2 = self.layernorm2(out1 + ffn_output)  # residual connection + norm

        return out2
