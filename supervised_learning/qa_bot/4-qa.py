#!/usr/bin/env python3
"""
Answers questions from multiple reference texts using semantic search.
"""

semantic_search = __import__('3-semantic_search').semantic_search
question_answer = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Answers questions interactively from multiple reference documents.

    Args:
        corpus_path (str): Path to the corpus folder containing reference texts.
    """
    while True:
        question = input("Q: ").strip()
        if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        # Find the most relevant document to the question
        reference = semantic_search(corpus_path, question)

        # Get the best answer from that document
        answer = question_answer(question, reference)

        if answer:
            print("A:", answer)
        else:
            print("A: Sorry, I do not understand your question.")
