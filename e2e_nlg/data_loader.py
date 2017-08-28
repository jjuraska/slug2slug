import sys
import os
import io
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize


def load_training_data(data_trainset, data_devset):
    # read the training data from file
    data_frame_train = pd.read_csv(data_trainset, header=0, encoding='latin1')  # names=['mr', 'ref']
    x_train = data_frame_train.mr.tolist()
    y_train = data_frame_train.ref.tolist()

    # read the development data from file
    data_frame_dev = pd.read_csv(data_devset, header=0, encoding='latin1')      # names=['mr', 'ref']
    x_dev = data_frame_dev.mr.tolist()
    y_dev = data_frame_dev.ref.tolist()

    # parse the utterances into lists of words
    y_train = [preprocess_utterance(y) for y in y_train]
    y_dev = [preprocess_utterance(y) for y in y_dev]

    # produce sequences of extracted words from the meaning representations (MRs) in the trainset
    x_train_seq = []
    for mr in x_train:
        row_list = []
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            row_list.extend([slot_word.lower() for slot_word in slot.split()])
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            row_list.extend([value_word.lower() for value_word in value.split()])

        x_train_seq.append(row_list)

    # produce sequences of extracted words from the meaning representations (MRs) in the devset
    x_dev_seq = []
    for mr in x_dev:
        row_list = []
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            row_list.extend([slot_word.lower() for slot_word in slot.split()])
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            row_list.extend([value_word.lower() for value_word in value.split()])

        x_dev_seq.append(row_list)

    with io.open('data/training_source.txt', 'w', encoding='utf8') as f_x_train:
        for line in x_train_seq:
            f_x_train.write('{}\n'.format(' '.join(line)))

    with io.open('data/training_target.txt', 'w', encoding='utf8') as f_y_train:
        for line in y_train:
            f_y_train.write('{}\n'.format(' '.join(line)))

    with io.open('data/dev_source.txt', 'w', encoding='utf8') as f_x_dev:
        for line in x_dev_seq:
            f_x_dev.write('{}\n'.format(' '.join(line)))

    with io.open('data/dev_target.txt', 'w', encoding='utf8') as f_y_dev:
        for line in y_dev:
            f_y_dev.write('{}\n'.format(' '.join(line)))


def load_test_data(data_test):
    # read the test data from file
    data_frame_test = pd.read_csv(data_test, header=0, encoding='latin1')  # names=['mr', 'ref']
    x_test = data_frame_test.mr.tolist()
    y_test = data_frame_test.ref.tolist()

    slots_with_proper_nouns = ['name', 'near', 'area', 'food']
    vocab_proper_nouns = set()

    # produce sequences of extracted words from the meaning representations (MRs) in the testset
    x_test_seq = []
    for mr in x_test:
        row_list = []
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            row_list.extend([slot_word.lower() for slot_word in slot.split()])
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            row_list.extend([value_word.lower() for value_word in value.split()])
            # store proper noun values (for retrieval in postprocessing)
            if slot in slots_with_proper_nouns and value[0].isupper():
                vocab_proper_nouns.add(value)

        x_test_seq.append(row_list)

    with io.open('data/test_source.txt', 'w', encoding='utf8') as f_x_test:
        for line in x_test_seq:
            f_x_test.write('{}\n'.format(' '.join(line)))

    with io.open('data/test_target.txt', 'w', encoding='utf8') as f_y_test:
        for line in y_test:
            f_y_test.write(line + '\n')

    with io.open('data/vocab_proper_nouns.txt', 'w', encoding='utf8') as f_vocab:
        for value in vocab_proper_nouns:
            f_vocab.write(value + '\n')


def preprocess_utterance(utterance):
    return word_tokenize(utterance.lower())


def delex_data(mrs, sentences, update_data_source=False, specific_slots=None, split=True):
    if specific_slots is not None:
        delex_slots = specific_slots
    else:
        delex_slots = ['name', 'food', 'near']

    for x, mr in enumerate(mrs):
        if split:
            sentence = ' '.join(sentences[x])
        else:
            sentence = sentences[x].lower()

        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            if slot in delex_slots:
                value = slot_value[sep_idx + 1:-1].strip()
                sentence = sentence.replace(value.lower(), '&slot_val_{0}&'.format(slot))
                mr = mr.replace(value, '&slot_val_{0}&'.format(slot))

        if update_data_source:
            if split:
                sentences[x] = sentence.split()
            else:
                sentences[x] = sentence
            mrs[x] = mr

        if not split:
            return sentence
