#!/usr/bin/env python3
"""
Transformer Decoder Block
"""
import tensorflow as tf
from tensorflow.keras import layers

# Import MultiHeadAttention
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(layers.Layer):
    """Decoder Block for Transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        Args:
            dm: int, model dimensionality
            h: int, number of attention heads
            hidden: int, number of hidden units in fully connected layer
            drop_rate: float, dropout rate
        """
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)   # masked multi-head attention
        self.mha2 = MultiHeadAttention(dm, h)   # encoder-decoder attention

        self.dense_hidden = layers.Dense(hidden, activation="relu")
        self.dense_output = layers.Dense(dm)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)
        self.dropout3 = layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for decoder block
        Args:
            x: tensor of shape (batch, target_seq_len, dm)
            encoder_output: tensor of shape (batch, input_seq_len, dm)
            training: boolean, whether in training mode
            look_ahead_mask: mask for the first MHA
            padding_mask: mask for the second MHA
        Returns:
            tensor of shape (batch, target_seq_len, dm)
        """
        # 1st Multi-head attention (masked self-attention)
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # 2nd Multi-head attention (encoder-decoder attention)
        attn2, _ = self.mha2(out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Feed-forward network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)

        out3 = self.layernorm3(out2 + ffn_output)

        return out3
