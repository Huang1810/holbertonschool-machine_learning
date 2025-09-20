#!/usr/bin/env python3
"""
Cumulative BLEU score
"""
import math


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence
    Args:
        references: list of reference translations (list of list of words)
        sentence: list of words in the proposed sentence
        n: size of the largest n-gram to use for evaluation
    Returns:
        The cumulative n-gram BLEU score
    """
    precisions = []

    for k in range(1, n + 1):
        # Candidate n-grams
        cand_ngrams = []
        for i in range(len(sentence) - k + 1):
            cand_ngrams.append(tuple(sentence[i:i + k]))

        cand_counts = {}
        for ng in cand_ngrams:
            cand_counts[ng] = cand_counts.get(ng, 0) + 1

        # Clip counts using references
        clipped_count = 0
        for ng, count in cand_counts.items():
            max_ref_count = 0
            for ref in references:
                ref_ngrams = []
                for i in range(len(ref) - k + 1):
                    ref_ngrams.append(tuple(ref[i:i + k]))
                ref_count = ref_ngrams.count(ng)
                max_ref_count = max(max_ref_count, ref_count)
            clipped_count += min(count, max_ref_count)

        precision = clipped_count / len(cand_ngrams) if cand_ngrams else 0
        precisions.append(precision)

    # Brevity penalty
    c = len(sentence)
    ref_lens = [len(r) for r in references]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - c), r))
    if c > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / c)

    # Geometric mean of precisions
    if min(precisions) == 0:
        geo_mean = 0
    else:
        geo_mean = math.exp(sum((1/n) * math.log(p) for p in precisions))

    bleu = bp * geo_mean
    return bleu
