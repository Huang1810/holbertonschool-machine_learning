#!/usr/bin/env python3
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    Args:
        max_seq_len: maximum sequence length (int)
        dm: model depth (int)

    Returns:
        numpy.ndarray of shape (max_seq_len, dm) with positional encodings
    """
    # Create position indices (max_seq_len, 1)
    pos = np.arange(max_seq_len)[:, np.newaxis]

    # Create dimension indices (1, dm)
    i = np.arange(dm)[np.newaxis, :]

    # Compute the angle rates
    angle_rates = pos / np.power(10000, (2 * (i // 2)) / np.float32(dm))

    # Apply sin to even indices, cos to odd indices
    PE = np.zeros((max_seq_len, dm))
    PE[:, 0::2] = np.sin(angle_rates[:, 0::2])
    PE[:, 1::2] = np.cos(angle_rates[:, 1::2])

    return PE
