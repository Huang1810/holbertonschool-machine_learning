#!/usr/bin/env python3
"""
Interactive question-answering loop using the question_answer() function.
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions interactively from a single reference text.
    
    Args:
        reference (str): The text document to use as a knowledge base.
    """
    while True:
        question = input("Q: ").strip()

        # Exit conditions (case-insensitive)
        if question.lower() in ('exit', 'quit', 'goodbye', 'bye'):
            print("A: Goodbye")
            break

        # Get answer from the QA model
        answer = question_answer(question, reference)

        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
