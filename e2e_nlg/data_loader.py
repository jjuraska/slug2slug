import sys
import os
import io
import json
import copy
import pandas as pd
import numpy as np
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
import itertools

def load_training_data(data_trainset, data_devset, input_concat=False):
    # read the training data from file
    data_frame_train = pd.read_csv(data_trainset, header=0, encoding='utf8')    # names=['mr', 'ref']
    x_train = data_frame_train.mr.tolist()
    y_train = data_frame_train.ref.tolist()

    # read the development data from file
    data_frame_dev = pd.read_csv(data_devset, header=0, encoding='utf8')        # names=['mr', 'ref']
    x_dev = data_frame_dev.mr.tolist()
    y_dev = data_frame_dev.ref.tolist()

    # parse the utterances into lists of words
    y_train = [preprocess_utterance(y) for y in y_train]
    y_dev = [preprocess_utterance(y) for y in y_dev]

    # produce sequences of extracted words from the meaning representations (MRs) in the trainset
    x_train_seq = []
    for i, mr in enumerate(x_train):
        mr_dict = OrderedDict()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()

            mr_dict[slot.lower()] = value.lower()

        y_train[i] = delex_sample(mr_dict, y_train[i], input_concat=True)

        # convert the dictionary to a list
        x_train_seq.append([])
        for key, val in mr_dict.items():
            x_train_seq[i].extend([key, val])

        if input_concat:
            x_train_seq[i].append('&stop&')


    # produce sequences of extracted words from the meaning representations (MRs) in the devset
    x_dev_seq = []
    for i, mr in enumerate(x_dev):
        mr_dict = OrderedDict()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()

            mr_dict[slot.lower()] = value.lower()

        y_dev[i] = delex_sample(mr_dict, y_dev[i], input_concat=True)

        # convert the dictionary to a list
        x_dev_seq.append([])
        for key, val in mr_dict.items():
            x_dev_seq[i].extend([key, val])

        if input_concat:
            x_dev_seq[i].append('&stop&')

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


def load_test_data(data_testset, input_concat=False):
    # read the test data from file
    data_frame_test = pd.read_csv(data_testset, header=0, encoding='utf8')  # names=['mr', 'ref']
    x_test = data_frame_test.mr.tolist()
    y_test = data_frame_test.ref.tolist()

    slots_with_proper_nouns = ['name', 'near', 'area', 'food']
    vocab_proper_nouns = set()

    # produce sequences of extracted words from the meaning representations (MRs) in the testset
    x_test_seq = []
    x_test_dict = []
    for i, mr in enumerate(x_test):
        mr_dict = OrderedDict()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()

            # store proper noun values (for retrieval in postprocessing)
            if slot in slots_with_proper_nouns and value[0].isupper():
                vocab_proper_nouns.add(value)

            mr_dict[slot.lower()] = value.lower()

        # build the MR dictionary
        x_test_dict.append(copy.deepcopy(mr_dict))

        delex_sample(mr_dict, y_test[i], mr_only=True, input_concat=True)

        # convert the dictionary to a list
        x_test_seq.append([])
        for key, val in mr_dict.items():
            x_test_seq[i].extend([key, val])

        if input_concat:
            x_test_seq[i].append('&stop&')

    with io.open('data/test_source.txt', 'w', encoding='utf8') as f_x_test:
        for line in x_test_seq:
            f_x_test.write('{}\n'.format(' '.join(line)))

    with io.open('data/test_source_dict.json', 'w', encoding='utf8') as f_x_test_dict:
        json.dump(x_test_dict, f_x_test_dict)

    with io.open('data/test_target.txt', 'w', encoding='utf8') as f_y_test:
        for line in y_test:
            f_y_test.write(line + '\n')

    # vocabulary of proper nouns to be used for capitalization in postprocessing
    with io.open('data/vocab_proper_nouns.txt', 'w', encoding='utf8') as f_vocab:
        for value in vocab_proper_nouns:
            f_vocab.write(value + '\n')

    # reference file for calculating metrics for test predictions
    with io.open('metrics/test_references.txt', 'w', encoding='utf8') as f_y_test:
        for i, line in enumerate(y_test):
            if i > 0 and x_test[i] != x_test[i - 1]:
                f_y_test.write('\n')
            f_y_test.write(line + '\n')


def preprocess_utterance(utterance):
    return word_tokenize(utterance.lower())


def delex_sample(mr, utterance, slots_to_delex=None, mr_only=False, input_concat=False):
    '''
    Delexicalize a single sample (MR and the corresponding utterance).
    By default, the slots 'name' and 'near' are delexicalized.
    All fields: name, near, area, food, customer rating, familyFriendly, eatType, priceRange
    '''
    vowels = 'aeiou'

    if slots_to_delex is not None:
        delex_slots = slots_to_delex
    else:
        delex_slots = ['name', 'near', 'food']

    if not mr_only:
        utterance = ' '.join(utterance)
    mr_update = {}

    for slot, value in mr.items():
        if slot in delex_slots:
            placeholder = '&slot_'
            if value[0].lower() in vowels:
                placeholder += 'vow_'
            else:
                placeholder += 'con_'

            if slot == 'food':
                if 'food' not in value.lower():
                    placeholder += 'cuisine_'
            placeholder += (slot + '&')

            if not mr_only:
                utterance = utterance.replace(value, placeholder)
            mr_update[slot] = placeholder
        else:
            if input_concat:
                mr_update[slot] = value.replace(' ', '_')

    for slot, new_value in mr_update.items():
        mr[slot] = new_value

    if not mr_only:
        return utterance.split()