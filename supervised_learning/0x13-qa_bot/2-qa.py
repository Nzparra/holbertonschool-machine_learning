#!/usr/bin/env python3
""" answers questions from a reference text """

from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf


def question_answer(question, reference):
    """
    question: string containing the question to answer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad",
        return_dict=True)
    inputs = tokenizer(question, reference,
                       add_special_tokens=True, return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    output = model(inputs)
    answer_start = tf.argmax(
        output.start_logits, axis=1
    ).numpy()[0]
    answer_end = (
            tf.argmax(output.end_logits, axis=1) + 1
    ).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def answer_loop(reference):
    """ answers questions from a reference text """
    while 1:
        question = input("Q: ")

        if question.lower() in ['exit', 'goodbye', 'bye']:
            print("A: Goodbye")
            exit(0)
        else:
            answer = question_answer(question, reference)
            if answer not in ['[CLS]']:
                print("A:", answer)
            else:
                print("A:", 'Sorry, I do not understand your question.')
