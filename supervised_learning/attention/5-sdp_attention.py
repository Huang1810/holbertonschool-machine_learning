#!/usr/bin/env python3
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention

    Args:
        Q: (..., seq_len_q, dk) query tensor
        K: (..., seq_len_v, dk) key tensor
        V: (..., seq_len_v, dv) value tensor
        mask: optional tensor broadcastable to (..., seq_len_q, seq_len_v)

    Returns:
        output: (..., seq_len_q, dv)
        weights: (..., seq_len_q, seq_len_v)
    """
    # Step 1: Calculate raw attention scores
    matmul_qk = tf.matmul(Q, K, transpose_b=True)  # (..., seq_len_q, seq_len_v)

    # Step 2: Scale scores by sqrt(dk)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Step 3: Apply mask (if any)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Step 4: Softmax across last axis (seq_len_v)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Step 5: Multiply by V to get output
    output = tf.matmul(weights, V)  # (..., seq_len_q, dv)

    return output, weights
