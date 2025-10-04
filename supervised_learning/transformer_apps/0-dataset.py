#!/usr/bin/env python3
"""
Dataset class for Machine Translation (Portuguese → English)
using TED HRLR dataset and pretrained BERT tokenizers
"""

import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    Loads and prepares the TED HRLR Portuguese-English dataset
    with pretrained tokenizers.
    """

    def __init__(self):
        """
        Constructor: loads dataset splits and tokenizers.
        """
        # Load dataset splits as supervised (pt, en)
        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="train",
            as_supervised=True
        )
        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="validation",
            as_supervised=True
        )

        # Create tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Loads pretrained tokenizers for Portuguese and English.
        Args:
            data: tf.data.Dataset of (pt, en) pairs
        Returns:
            tokenizer_pt, tokenizer_en
        """
        tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            use_fast=True
        )
        tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            use_fast=True
        )

        return tokenizer_pt, tokenizer_en
