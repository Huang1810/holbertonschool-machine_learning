#!/usr/bin/env python3
"""
Transformer Decoder
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Import helpers
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(layers.Layer):
    """Decoder for Transformer"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor
        Args:
            N: int, number of decoder blocks
            dm: int, model dimensionality
            h: int, number of attention heads
            hidden: int, number of hidden units in fully connected layer
            target_vocab: int, size of target vocabulary
            max_seq_len: int, maximum sequence length
            drop_rate: float, dropout rate
        """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for decoder
        Args:
            x: tensor of shape (batch, target_seq_len)
            encoder_output: tensor of shape (batch, input_seq_len, dm)
            training: boolean, whether in training mode
            look_ahead_mask: mask for first multi-head attention layer
            padding_mask: mask for second multi-head attention layer
        Returns:
            tensor of shape (batch, target_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]

        # Token embeddings
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encodings
        x += self.positional_encoding[:seq_len]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each decoder block
        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask, padding_mask)

        return x
