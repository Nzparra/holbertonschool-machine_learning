#!/usr/bin/env python3
"""
creating semantic similarity program
"""

import os
from transformers import BertTokenizer, TFBertForQuestionAnswering
import tensorflow as tf
import tensorflow_hub as hub


def modelResult(model, tokenizer, sent1, sent2, reverse=0):
    """
    Return: list with [avg, ans, text] if reverse = 0
            list with [avg, text] if reverse = 1
    """
    if reverse == 0:
        input_dict = tokenizer(sent1.lower(),
                               sent2.lower(),
                               return_tensors='tf')
    else:
        input_dict = tokenizer(sent2.lower(),
                               sent1.lower(),
                               return_tensors='tf')
    outputs = model(input_dict)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    x = input_dict["input_ids"].numpy()[0]
    all_tokens = tokenizer.convert_ids_to_tokens(x)
    begin = tf.math.argmax(start_logits, 1)[0]
    last = tf.math.argmax(end_logits, 1)[0] + 1
    answer = ' '.join(all_tokens[begin: last])
    first = tf.reduce_max(start_logits, 1)[0]
    last = tf.reduce_max(end_logits, 1)[0] + 1
    avg = (first + last) / 2
    if len(answer) > 0:
        if reverse == 0:
            return([avg, answer, sent2])
        else:
            return([avg, sent2])
    return None


def semantic_search(corpus_path, sentence):
    """
    Returns: a string containing the answer
    * If no answer is found, return None
    """
    files = os.listdir(corpus_path)
    files = [elem for elem in files if '.md' in elem]
    all_text = []
    for file in files:
        with open('ZendeskArticles/' + file, 'r', encoding='UTF-8') as f:
            f_line = f.read()
        all_text.append(f_line)
    url = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(url)
    model = TFBertForQuestionAnswering.from_pretrained(url,
                                                       return_dict=True)
    result = []
    for i in range(len(all_text)):
        r = modelResult(model, tokenizer, sentence, all_text[i], 0)
        if type(r) is list:
            r.append(files[i])
            result.append(r)
    aux_sort = sorted(result, key=lambda x: x[0], reverse=True)
    best_5 = aux_sort[:5]
    new_scores = []
    for elem in best_5:
        r = modelResult(model, tokenizer, sentence, elem[2], 1)
        if type(r) is list:
            r.append(elem[1])
            r.append(elem[3])
            new_scores.append(r)
    aux_sort = sorted(new_scores, key=lambda x: abs(x[0]), reverse=True)
    return aux_sort[0][1]


def question_answer(question, reference):
    """
    Returns: a string containing the answer
    """
    param = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(param)
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    question_token = tokenizer.tokenize(question)
    reference_token = tokenizer.tokenize(reference)
    sep1 = ['[CLS]']
    sep2 = ['[SEP]']
    tokens = sep1 + question_token + sep2 + reference_token + sep2
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_token) + 1) +\
                     [1] * (len(reference_token) + 1)
    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids,
                                                      input_mask,
                                                      input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    if len(answer) > 1:
        return answer
    else:
        return None
    return answer


def qa_bot(corpus_path):
    """
    * corpus_path is the path to the corpus of reference documents
    Return response
    """
    EXIT = ['exit', 'quit', 'goodbye', 'bye']
    while(1):
        user_question = input("Q: ")
        if user_question.lower() in EXIT:
            print("A: Goodbye")
            break
        else:
            text = semantic_search(corpus_path, user_question)
            response = question_answer(user_question, text)
            print("A: {}".format(response))
    exit(0)
