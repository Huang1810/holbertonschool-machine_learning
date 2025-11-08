#!/usr/bin/env python3
"""
Interactive Q&A loop.
Prompts the user with 'Q:' and responds with 'A:'.
If the user inputs exit, quit, goodbye, or bye (case-insensitive), exits politely.
"""

def main():
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit", "goodbye", "bye"):
            print("A: Goodbye")
            break
        else:
            print("A:")

if __name__ == "__main__":
    main()
