#!/usr/bin/env python3
"""
n-gram BLEU score
"""
import math


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence
    Args:
        references: list of reference translations (list of list of words)
        sentence: list of words in the proposed sentence
        n: size of the n-gram to use for evaluation
    Returns:
        The n-gram BLEU score
    """
    # Create n-grams for candidate
    cand_ngrams = []
    for i in range(len(sentence) - n + 1):
        cand_ngrams.append(tuple(sentence[i:i + n]))

    # Count candidate n-grams
    cand_counts = {}
    for ng in cand_ngrams:
        cand_counts[ng] = cand_counts.get(ng, 0) + 1

    # Count reference n-grams
    clipped_count = 0
    for ng, count in cand_counts.items():
        max_ref_count = 0
        for ref in references:
            ref_ngrams = []
            for i in range(len(ref) - n + 1):
                ref_ngrams.append(tuple(ref[i:i + n]))
            ref_count = ref_ngrams.count(ng)
            max_ref_count = max(max_ref_count, ref_count)
        clipped_count += min(count, max_ref_count)

    precision = clipped_count / len(cand_ngrams) if cand_ngrams else 0

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
