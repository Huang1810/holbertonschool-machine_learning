#!/usr/bin/env python3
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        # GRU layer
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

        # Dense layer for output vocabulary
        self.F = tf.keras.layers.Dense(vocab)

        # Self-Attention
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        x: (batch, 1) - previous word index
        s_prev: (batch, units) - previous decoder hidden state
        hidden_states: (batch, input_seq_len, units) - encoder outputs
        Returns:
            y: (batch, vocab) - prediction over vocabulary
            s: (batch, units) - new hidden state
        """
        # Compute context vector using attention
        context, _ = self.attention(s_prev, hidden_states)  # (batch, units)

        # Embed input word
        x = self.embedding(x)  # (batch, 1, embedding_dim)

        # Concatenate context with embedded input word
        context = tf.expand_dims(context, 1)  # (batch, 1, units)
        x = tf.concat([context, x], axis=-1)  # (batch, 1, units+embedding)

        # Pass through GRU
        output, s = self.gru(x, initial_state=s_prev)  # output: (batch, 1, units)

        # Squeeze out time dimension
        output = tf.reshape(output, (-1, output.shape[2]))  # (batch, units)

        # Vocabulary prediction
        y = self.F(output)  # (batch, vocab)

        return y, s
