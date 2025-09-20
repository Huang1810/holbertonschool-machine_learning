#!/usr/bin/env python3
"""
TF-IDF function
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix

    Args:
        sentences: list of sentences (strings)
        vocab: list of vocabulary words to use (optional)

    Returns:
        embeddings: numpy.ndarray of shape (len(sentences), len(vocab))
        features: numpy.ndarray of shape (len(vocab),)
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences)
    return embeddings.toarray(), np.array(vectorizer.get_feature_names_out())
