#!/usr/bin/env python3
"""
Defines a Transformer model adapted for Portuguese→English translation
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, Dropout, MultiHeadAttention


class PositionalEncoding(tf.keras.layers.Layer):
    """Adds sinusoidal positional encoding to embeddings."""
    def __init__(self, max_len, dm):
        super().__init__()
        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(dm, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / dm)
        angle_rads = pos * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


def encoder_block(dm, h, hidden, dropout_rate=0.1):
    inputs = tf.keras.Input(shape=(None, dm))
    attn_output = MultiHeadAttention(num_heads=h, key_dim=dm)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(hidden, activation='relu')(out1)
    ffn_output = Dense(dm)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return tf.keras.Model(inputs=inputs, outputs=out2)


def decoder_block(dm, h, hidden, dropout_rate=0.1):
    inputs = tf.keras.Input(shape=(None, dm))
    enc_output = tf.keras.Input(shape=(None, dm))

    # Masked self-attention
    attn1 = MultiHeadAttention(num_heads=h, key_dim=dm)(inputs, inputs)
    attn1 = Dropout(dropout_rate)(attn1)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn1)

    # Encoder-decoder attention
    attn2 = MultiHeadAttention(num_heads=h, key_dim=dm)(out1, enc_output)
    attn2 = Dropout(dropout_rate)(attn2)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + attn2)

    # Feed-forward network
    ffn_output = Dense(hidden, activation='relu')(out2)
    ffn_output = Dense(dm)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out3 = LayerNormalization(epsilon=1e-6)(out2 + ffn_output)

    return tf.keras.Model([inputs, enc_output], out3)


class Transformer(tf.keras.Model):
    """Full Transformer model for sequence-to-sequence translation."""
    def __init__(self, N, dm, h, hidden, input_vocab_size, target_vocab_size, max_len, dropout_rate=0.1):
        super().__init__()
        self.dm = dm
        self.N = N
        self.input_embedding = Embedding(input_vocab_size, dm)
        self.target_embedding = Embedding(target_vocab_size, dm)
        self.pos_encoding_input = PositionalEncoding(max_len, dm)
        self.pos_encoding_target = PositionalEncoding(max_len, dm)
        self.encoder_blocks = [encoder_block(dm, h, hidden, dropout_rate) for _ in range(N)]
        self.decoder_blocks = [decoder_block(dm, h, hidden, dropout_rate) for _ in range(N)]
        self.final_layer = Dense(target_vocab_size)

    def call(self, inputs, target, training, encoder_mask=None, look_ahead_mask=None, decoder_mask=None):
        x = self.input_embedding(inputs)
        x = self.pos_encoding_input(x)
        for enc_block in self.encoder_blocks:
            x = enc_block(x)

        y = self.target_embedding(target)
        y = self.pos_encoding_target(y)
        for dec_block in self.decoder_blocks:
            y = dec_block([y, x])

        final_output = self.final_layer(y)
        return final_output, None
