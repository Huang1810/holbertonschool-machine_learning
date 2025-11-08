#!/usr/bin/env python3
"""
Finds a text snippet within a reference document that answers a question
using a pretrained BERT QA model.
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Args:
        question (str): The question to answer.
        reference (str): The reference text.

    Returns:
        str or None: The answer string if found, otherwise None.
    """
    try:
        # Load pre-trained model and tokenizer
        model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
        tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        )

        # Tokenize the input question and reference text
        inputs = tokenizer.encode_plus(
            question, reference, add_special_tokens=True, return_tensors="tf"
        )

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]

        # Run inference using the BERT QA model
        outputs = model([input_ids, token_type_ids])
        start_logits = outputs[0]
        end_logits = outputs[1]

        # Get the most probable start and end positions
        start_index = tf.argmax(start_logits, axis=1).numpy()[0]
        end_index = tf.argmax(end_logits, axis=1).numpy()[0] + 1

        # Convert tokens back to words
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy())
        answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index])

        # Handle invalid or empty results
        if not answer or answer.strip() in ["[CLS]", "[SEP]"]:
            return None

        return answer.strip()

    except Exception:
        return None
