#!/usr/bin/env python3
"""
Performs semantic search on a corpus of documents using the Universal Sentence Encoder.
"""

import os
import tensorflow_hub as hub
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of reference documents.

    Args:
        corpus_path (str): The path to the directory containing text documents.
        sentence (str): The query sentence to search for semantically.

    Returns:
        str: The text content of the document most similar to the sentence.
    """
    # Load Universal Sentence Encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # Load all text documents from the corpus directory
    documents = []
    file_names = []

    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(text)
                file_names.append(filename)

    if not documents:
        raise ValueError("No documents found in corpus path.")

    # Encode the sentence and the corpus
    embeddings = embed([sentence] + documents)
    query_vec = embeddings[0]
    doc_vecs = embeddings[1:]

    # Compute cosine similarity between query and each document
    similarities = np.inner(query_vec, doc_vecs)

    # Get index of most similar document
    most_similar_idx = np.argmax(similarities)

    # Return the most relevant document text
    return documents[most_similar_idx]
