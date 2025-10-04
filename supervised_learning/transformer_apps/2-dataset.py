#!/usr/bin/env python3
"""
Minimal Dataset class for TED HRLR Portuguese→English translation
using pretrained BERT tokenizers and TensorFlow.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads, tokenizes, and encodes TED HRLR translation dataset
    (Portuguese → English) with pretrained BERT tokenizers.
    """

    def __init__(self):
        # Load dataset splits as supervised (pt, en)
        data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='train', as_supervised=True)
        data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='validation', as_supervised=True)

        # Initialize pretrained tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset()

        # Map datasets through tf_encode
        self.data_train = data_train.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        self.data_valid = data_valid.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)

    def tokenize_dataset(self):
        """Load pretrained BERT tokenizers for Portuguese and English"""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a (pt, en) pair into integer token sequences"""
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        # Tokenize sentences without retraining
        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence, add_special_tokens=True)
        en_tokens = self.tokenizer_en.encode(
            en_sentence, add_special_tokens=True)

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for the encode method"""
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Set 1D shape
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
