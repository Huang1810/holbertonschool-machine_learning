#!/usr/bin/env python3
"""
Transformer Network
"""
import tensorflow as tf
from tensorflow.keras import Model, layers

# Import Encoder and Decoder
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(Model):
    """Transformer Network"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor
        Args:
            N: int, number of encoder/decoder blocks
            dm: int, model dimensionality
            h: int, number of heads
            hidden: int, hidden units in fully connected layers
            input_vocab: int, size of input vocabulary
            target_vocab: int, size of target vocabulary
            max_seq_input: int, max sequence length for input
            max_seq_target: int, max sequence length for target
            drop_rate: float, dropout rate
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """
        Forward pass for transformer
        Args:
            inputs: tensor of shape (batch, input_seq_len)
            target: tensor of shape (batch, target_seq_len)
            training: boolean, training mode
            encoder_mask: padding mask for encoder
            look_ahead_mask: look ahead mask for decoder
            decoder_mask: padding mask for decoder
        Returns:
            tensor of shape (batch, target_seq_len, target_vocab)
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training, look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)
        return final_output
