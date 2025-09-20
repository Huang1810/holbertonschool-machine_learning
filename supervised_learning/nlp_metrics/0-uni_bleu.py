#!/usr/bin/env python3
"""
Unigram BLEU score
"""
import math


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    Args:
        references: list of reference translations (list of list of words)
        sentence: list of words in the proposed sentence
    Returns:
        The unigram BLEU score
    """
    # Count candidate word occurrences
    word_counts = {}
    for w in sentence:
        word_counts[w] = word_counts.get(w, 0) + 1

    # Clip counts using references
    clipped_count = 0
    for word, count in word_counts.items():
        max_ref_count = 0
        for ref in references:
            max_ref_count = max(max_ref_count, ref.count(word))
        clipped_count += min(count, max_ref_count)

    precision = clipped_count / len(sentence)

    # Brevity penalty
    c = len(sentence)
    ref_lens = [len(r) for r in references]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - c), r))
    if c > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / c)

    bleu = bp * precision
    return bleu
