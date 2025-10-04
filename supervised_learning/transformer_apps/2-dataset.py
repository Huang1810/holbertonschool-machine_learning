#!/usr/bin/env python3
"""
Dataset class for Machine Translation (Portuguese → English)
using TED HRLR dataset and pretrained BERT tokenizers
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    A class to load and prepare the TED HRLR translation dataset for machine
    translation from Portuguese to English.
    """

    def __init__(self):
        """
        Initializes the Dataset object and loads the training and validation
        datasets for Portuguese to English translation.
        """
        # Load the TED HRLR Portuguese to English translation dataset
        data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='train', as_supervised=True)
        data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='validation', as_supervised=True)

        # Initialize pretrained tokenizers (no retraining)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(data_train)

        # Apply encoding to the datasets
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Loads pretrained tokenizers for Portuguese and English
        """
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
        """
        Encodes Portuguese and English sentences into tokens
        """
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        # Tokenize sentences
        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence, add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(
            en_sentence, add_special_tokens=False)

        # Add start/end tokens
        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode method
        """
        result_pt, result_en = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        # Set shape so TF knows it's 1D
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
