#!/usr/bin/env python3
"""
contains the semantic_search function
"""
import pathlib
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf
import operator


def semantic_score(sentence, reference):
    """
        return: string containing the answer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad",
        return_dict=True)
    inputs = tokenizer(sentence, reference, add_special_tokens=True,
                       return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    output = model(inputs)
    answer_start = tf.argmax(
        output.start_logits, axis=1
    ).numpy()[0]
    start = output.start_logits[:, answer_start].numpy()[0]
    print('start:', answer_start, start)
    answer_end = (
            tf.argmax(output.end_logits, axis=1) + 1
    ).numpy()[0]
    end = output.start_logits[:, answer_end].numpy()[0]
    print('end:', answer_end, end)
    return start, end


def semantic_search(corpus_path, sentence):
    """
        return: reference text of the document most similar to sentence
    """
    my_dict = {}
    for path in pathlib.Path(corpus_path).iterdir():
        if path.is_file():
            print('\n', '\n', '**********************\n', path)
            with open(path, 'rb') as f:
                reference = f.read().decode(errors='replace')
                start, end = semantic_score(sentence, reference)
                avg = (start + end) / 2
                print('avg:', avg)
                my_dict[path] = avg
                f.close()
    my_doc = max(my_dict.items(), key=operator.itemgetter(1))[0]
    with open(my_doc, 'rb') as f:
        reference = f.read().decode(errors='replace')
    return reference
