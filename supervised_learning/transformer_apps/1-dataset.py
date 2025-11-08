#!/usr/bin/env python3
"""
Dataset class for Portuguese-English translation
"""

import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    def __init__(self):
        # Load the Portuguese-English dataset from TFDS
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True, as_supervised=True
        )
        self.data_train = examples['train']
        self.data_valid = examples['validation']

        # Initialize subword tokenizers
        self.tokenizer_pt, self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in self.data_train),
            target_vocab_size=2**13
        ), tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in self.data_train),
            target_vocab_size=2**13
        )

        # Start and end token indices
        self.start_token_pt = self.tokenizer_pt.vocab_size
        self.end_token_pt = self.tokenizer_pt.vocab_size + 1
        self.start_token_en = self.tokenizer_en.vocab_size
        self.end_token_en = self.tokenizer_en.vocab_size + 1

    def encode(self, pt, en):
        """
        Encodes Portuguese and English sentences into token indices,
        adding start and end tokens.

        Args:
            pt (tf.Tensor): Portuguese sentence
            en (tf.Tensor): English sentence

        Returns:
            pt_tokens (np.ndarray): Tokenized Portuguese sentence
            en_tokens (np.ndarray): Tokenized English sentence
        """
        # Convert tf.Tensor to bytes and decode to string
        pt_str = pt.numpy().decode('utf-8')
        en_str = en.numpy().decode('utf-8')

        # Encode sentences using the subword tokenizer
        pt_tokens = self.tokenizer_pt.encode(pt_str)
        en_tokens = self.tokenizer_en.encode(en_str)

        # Add start and end tokens
        pt_tokens = [self.start_token_pt] + pt_tokens + [self.end_token_pt]
        en_tokens = [self.start_token_en] + en_tokens + [self.end_token_en]

        return np.array(pt_tokens), np.array(en_tokens)
