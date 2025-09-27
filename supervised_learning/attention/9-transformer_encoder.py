#!/usr/bin/env python3
"""
Transformer Encoder
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Import helpers
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(layers.Layer):
    """Encoder for Transformer"""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor
        Args:
            N: int, number of encoder blocks
            dm: int, model dimensionality
            h: int, number of attention heads
            hidden: int, number of hidden units in fully connected layer
            input_vocab: int, size of input vocabulary
            max_seq_len: int, maximum sequence length
            drop_rate: float, dropout rate
        """
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass for encoder
        Args:
            x: tensor of shape (batch, input_seq_len)
            training: boolean, whether in training mode
            mask: mask for multi-head attention
        Returns:
            tensor of shape (batch, input_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]

        # Token embeddings
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encodings
        x += self.positional_encoding[:seq_len]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each encoder block
        for block in self.blocks:
            x = block(x, training, mask)

        return x
