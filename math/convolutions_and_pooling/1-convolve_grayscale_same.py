#!/usr/bin/env python3
"""
Same Convolution (grayscale)
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    convolved = np.zeros((m, h, w))

    pad_width = ((0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2))
    padded_imgs = np.pad(
            images, pad_width=pad_width, mode="constant", constant_values=0
            )

    for i in range(h):
        for j in range(w):

            region = padded_imgs[:, i:(i + kh), j:(j + kw)]

            convolved[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved
