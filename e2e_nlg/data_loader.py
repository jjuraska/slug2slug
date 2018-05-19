import os
import io
import string
import json
import copy
import pandas as pd
import numpy as np
from collections import OrderedDict
import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re

import config


EMPH_TOKEN = config.EMPH_TOKEN
CONTRAST_TOKEN = config.CONTRAST_TOKEN
CONCESSION_TOKEN = config.CONCESSION_TOKEN


# TODO: redesign the data loading so as to be object-oriented
def load_training_data(data_trainset, data_devset, input_concat=False, generate_vocab=False):
    training_source_file = os.path.join(config.DATA_DIR, 'training_source.txt')
    training_target_file = os.path.join(config.DATA_DIR, 'training_target.txt')
    dev_source_file = os.path.join(config.DATA_DIR, 'dev_source.txt')
    dev_target_file = os.path.join(config.DATA_DIR, 'dev_target.txt')

    if os.path.isfile(training_source_file) and \
            os.path.isfile(training_target_file) and \
            os.path.isfile(dev_source_file) and \
            os.path.isfile(dev_target_file):
        print('Found existing input files. Skipping their generation.')
        return

    dataset = init_training_data(data_trainset, data_devset)
    dataset_name = dataset['dataset_name']
    x_train, y_train, x_dev, y_dev = dataset['data']
    slot_sep, val_sep, val_sep_closing = dataset['separators']

    # TODO: do the utterances still need to be parsed into lists of words?
    # parse the utterances into lists of words
    y_train = [preprocess_utterance(y) for y in y_train]
    y_dev = [preprocess_utterance(y) for y in y_dev]

    # produce sequences of extracted words from the meaning representations (MRs) in the trainset
    x_train_seq = []
    for i, mr in enumerate(x_train):
        slot_ctr = 0
        emph_idxs = set()
        mr_dict = OrderedDict()

        # extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_closing)

            if slot == EMPH_TOKEN:
                emph_idxs.add(slot_ctr)
            else:
                mr_dict[slot] = value
                slot_ctr += 1

        # delexicalize the MR and the utterance
        y_train[i] = delex_sample(mr_dict, y_train[i], dataset=dataset_name, input_concat=input_concat)

        slot_ctr = 0

        # convert the dictionary to a list
        x_train_seq.append([])
        for key, val in mr_dict.items():
            # insert the emphasis token where appropriate
            if slot_ctr in emph_idxs:
                x_train_seq[i].append(EMPH_TOKEN)

            if len(val) > 0:
                x_train_seq[i].extend([key] + val.split())
            else:
                x_train_seq[i].append(key)

            slot_ctr += 1

        if input_concat:
            # append a sequence-end token to be paired up with seq2seq's sequence-end token when concatenating
            x_train_seq[i].append('<STOP>')

    # produce sequences of extracted words from the meaning representations (MRs) in the devset
    x_dev_seq = []
    for i, mr in enumerate(x_dev):
        slot_ctr = 0
        emph_idxs = set()
        mr_dict = OrderedDict()

        # extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_closing)

            if slot == EMPH_TOKEN:
                emph_idxs.add(slot_ctr)
            else:
                mr_dict[slot] = value
                slot_ctr += 1

        # delexicalize the MR and the utterance
        y_dev[i] = delex_sample(mr_dict, y_dev[i], dataset=dataset_name, input_concat=input_concat)

        slot_ctr = 0

        # convert the dictionary to a list
        x_dev_seq.append([])
        for key, val in mr_dict.items():
            # insert the emphasis token where appropriate
            if slot_ctr in emph_idxs:
                x_dev_seq[i].append(EMPH_TOKEN)

            if len(val) > 0:
                x_dev_seq[i].extend([key] + val.split())
            else:
                x_dev_seq[i].append(key)

            slot_ctr += 1

        if input_concat:
            # append a sequence-end token to be paired up with seq2seq's sequence-end token when concatenating
            x_dev_seq[i].append('<STOP>')

    # Generate a vocabulary file if necessary
    if generate_vocab:
        generate_vocab_file(np.concatenate(x_train_seq + x_dev_seq + y_train + y_dev),
                            vocab_filename='vocab.lang_gen.tokens')
        # generate_vocab_file(np.concatenate(x_train_seq + x_dev_seq),
        #                     vocab_filename='vocab.lang_gen_multi_vocab.source')
        # generate_vocab_file(np.concatenate(y_train + y_dev),
        #                     vocab_filename='vocab.lang_gen_multi_vocab.target')

    with io.open(training_source_file, 'w', encoding='utf8') as f_x_train:
        for line in x_train_seq:
            f_x_train.write('{}\n'.format(' '.join(line)))

    with io.open(training_target_file, 'w', encoding='utf8') as f_y_train:
        for line in y_train:
            f_y_train.write('{}\n'.format(' '.join(line)))

    with io.open(dev_source_file, 'w', encoding='utf8') as f_x_dev:
        for line in x_dev_seq:
            f_x_dev.write('{}\n'.format(' '.join(line)))

    with io.open(dev_target_file, 'w', encoding='utf8') as f_y_dev:
        for line in y_dev:
            f_y_dev.write('{}\n'.format(' '.join(line)))

    return np.concatenate(x_train_seq + x_dev_seq + y_train + y_dev).flatten()


def load_test_data(data_testset, input_concat=False):
    test_source_file = os.path.join(config.DATA_DIR, 'test_source.txt')
    test_source_dict_file = os.path.join(config.DATA_DIR, 'test_source_dict.json')
    test_target_file = os.path.join(config.DATA_DIR, 'test_target.txt')
    test_reference_file = os.path.join(config.METRICS_DIR, 'test_references.txt')
    vocab_proper_nouns_file = os.path.join(config.DATA_DIR, 'vocab_proper_nouns.txt')

    dataset = init_test_data(data_testset)
    dataset_name = dataset['dataset_name']
    x_test, y_test = dataset['data']
    slot_sep, val_sep, val_sep_closing = dataset['separators']

    slots_with_proper_nouns = ['name', 'near', 'area', 'food']
    vocab_proper_nouns = set()

    # produce sequences of extracted words from the meaning representations (MRs) in the testset
    x_test_seq = []
    x_test_dict = []
    for i, mr in enumerate(x_test):
        slot_ctr = 0
        emph_idxs = set()
        mr_dict = OrderedDict()

        # extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_closing)

            # store proper noun values (for retrieval in postprocessing)
            if slot in slots_with_proper_nouns and len(value_orig) > 0 and value_orig[0].isupper():
                vocab_proper_nouns.add(value_orig)

            if slot == EMPH_TOKEN:
                emph_idxs.add(slot_ctr)
            else:
                mr_dict[slot] = value
                slot_ctr += 1

        # build the MR dictionary
        x_test_dict.append(copy.deepcopy(mr_dict))

        # delexicalize the MR
        delex_sample(mr_dict, dataset=dataset_name, mr_only=True, input_concat=input_concat)

        slot_ctr = 0

        # convert the dictionary to a list
        x_test_seq.append([])
        for key, val in mr_dict.items():
            # insert the emphasis token where appropriate
            if slot_ctr in emph_idxs:
                x_test_seq[i].append(EMPH_TOKEN)

            if len(val) > 0:
                x_test_seq[i].extend([key] + val.split())
            else:
                x_test_seq[i].append(key)

            slot_ctr += 1

        if input_concat:
            # append a sequence-end token to be paired up with seq2seq's sequence-end token when concatenating
            x_test_seq[i].append('<STOP>')

    with io.open(test_source_file, 'w', encoding='utf8') as f_x_test:
        for line in x_test_seq:
            f_x_test.write('{}\n'.format(' '.join(line)))

    with io.open(test_source_dict_file, 'w', encoding='utf8') as f_x_test_dict:
        json.dump(x_test_dict, f_x_test_dict)

    # vocabulary of proper nouns to be used for capitalization in postprocessing
    with io.open(vocab_proper_nouns_file, 'w', encoding='utf8') as f_vocab:
        for value in vocab_proper_nouns:
            f_vocab.write(value + '\n')

    if len(y_test) > 0:
        with io.open(test_target_file, 'w', encoding='utf8') as f_y_test:
            for line in y_test:
                f_y_test.write(line + '\n')

        # reference file for calculating metrics for test predictions
        with io.open(test_reference_file, 'w', encoding='utf8') as f_y_test:
            for i, line in enumerate(y_test):
                if i > 0 and x_test[i] != x_test[i - 1]:
                    f_y_test.write('\n')
                f_y_test.write(line + '\n')


def generate_vocab_file(token_sequences, vocab_filename, vocab_size=10000):
    vocab_file = os.path.join(config.DATA_DIR, vocab_filename)

    distr = FreqDist(token_sequences)
    vocab = distr.most_common(min(len(distr), vocab_size - 3))      # cap the vocabulary size

    vocab_with_reserved_tokens = ['<pad>', '<EOS>'] + list(map(lambda tup: tup[0], vocab)) + ['UNK']

    with io.open(vocab_file, 'w', encoding='utf8') as f_vocab:
        for token in vocab_with_reserved_tokens:
            f_vocab.write('{}\n'.format(token))


def get_vocabulary(token_sequences, vocab_size=10000):
    distr = FreqDist(token_sequences)
    vocab = distr.most_common(min(len(distr), vocab_size))          # cap the vocabulary size

    vocab_set = set(map(lambda tup: tup[0], vocab))

    return vocab_set


def tokenize_mr(mr, add_eos_token=True):
    '''
    Produces a (delexed) sequence of tokens from the input MR.
    '''

    slot_sep = ','
    val_sep = '['
    val_sep_closing = True
    
    mr_seq = []
    mr_dict = OrderedDict()

    # extract the slot-value pairs into a dictionary
    for slot_value in mr.split(slot_sep):
        slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
        mr_dict[slot] = value

    # make a copy of the dictionary for delexing
    mr_dict_delex = copy.deepcopy(mr_dict)

    # delexicalize the MR
    delex_sample(mr_dict_delex, mr_only=True)

    # convert the dictionary to a list
    for key, val in mr_dict_delex.items():
        if len(val) > 0:
            mr_seq.extend([key, val])
        else:
            mr_seq.append(key)

    # append the sequence-end token
    if add_eos_token:
        mr_seq.append('SEQUENCE_END')

    return mr_seq, mr_dict


def load_training_data_for_eval(data_trainset, data_model_outputs_train, vocab_size, max_input_seq_len, max_output_seq_len, delex=False):
    dataset_name = ''
    slot_sep = ''
    val_sep = ''
    val_sep_closing = False

    if '/rest_e2e/' in data_trainset or '\\rest_e2e\\' in data_trainset:
        x_train, y_train_1 = read_rest_e2e_dataset_train(data_trainset)
        y_train_2 = read_predictions(data_model_outputs_train)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_closing = True
    elif '/tv/' in data_trainset or '\\tv\\' in data_trainset:
        x_train, y_train_1, y_train_2 = read_tv_dataset_train(data_trainset)
        if data_model_outputs_train is not None:
            y_train_2 = read_predictions(data_model_outputs_train)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
    elif '/laptop/' in data_trainset or '\\laptop\\' in data_trainset:
        x_train, y_train_1, y_train_2 = read_laptop_dataset_train(data_trainset)
        if data_model_outputs_train is not None:
            y_train_2 = read_predictions(data_model_outputs_train)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
    else:
        raise FileNotFoundError

    # parse the utterances into lists of words
    y_train_1 = [preprocess_utterance(y) for y in y_train_1]
    y_train_2 = [preprocess_utterance(y) for y in y_train_2]
    

    # produce sequences of extracted words from the meaning representations (MRs) in the trainset
    x_train_seq = []
    for i, mr in enumerate(x_train):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value

        if delex == True:
            # delexicalize the MR and the utterance
            y_train_1[i] = delex_sample(mr_dict, y_train_1[i], dataset=dataset_name, utterance_only=True)
            y_train_2[i] = delex_sample(mr_dict, y_train_2[i], dataset=dataset_name)

        # convert the dictionary to a list
        x_train_seq.append([])
        for key, val in mr_dict.items():
            if len(val) > 0:
                x_train_seq[i].extend([key, val])
            else:
                x_train_seq[i].append(key)


    # create source vocabulary
    if os.path.isfile('data/eval_vocab_source.json'):
        with io.open('data/eval_vocab_source.json', 'r', encoding='utf8') as f_x_vocab:
            x_vocab = json.load(f_x_vocab)
    else:
        x_distr = FreqDist([x_token for x in x_train_seq for x_token in x])
        x_vocab = x_distr.most_common(min(len(x_distr), vocab_size - 2))        # cap the vocabulary size
        with io.open('data/eval_vocab_source.json', 'w', encoding='utf8') as f_x_vocab:
            json.dump(x_vocab, f_x_vocab, ensure_ascii=False)

    x_idx2word = [word[0] for word in x_vocab]
    x_idx2word.insert(0, '<PADDING>')
    x_idx2word.append('<NA>')
    x_word2idx = {word: idx for idx, word in enumerate(x_idx2word)}

    # create target vocabulary
    if os.path.isfile('data/eval_vocab_target.json'):
        with io.open('data/eval_vocab_target.json', 'r', encoding='utf8') as f_y_vocab:
            y_vocab = json.load(f_y_vocab)
    else:
        y_distr = FreqDist([y_token for y in y_train_1 for y_token in y] + [y_token for y in y_train_2 for y_token in y])
        y_vocab = y_distr.most_common(min(len(y_distr), vocab_size - 2))        # cap the vocabulary size
        with io.open('data/eval_vocab_target.json', 'w', encoding='utf8') as f_y_vocab:
            json.dump(y_vocab, f_y_vocab, ensure_ascii=False)

    y_idx2word = [word[0] for word in y_vocab]
    y_idx2word.insert(0, '<PADDING>')
    y_idx2word.append('<NA>')
    y_word2idx = {token: idx for idx, token in enumerate(y_idx2word)}


    # produce sequences of indexes from the MRs in the training set
    x_train_enc = token_seq_to_idx_seq(x_train_seq, x_word2idx, max_input_seq_len)

    # produce sequences of indexes from the utterances in the training set
    y_train_1_enc = token_seq_to_idx_seq(y_train_1, y_word2idx, max_output_seq_len)

    # produce sequences of indexes from the utterances in the training set
    y_train_2_enc = token_seq_to_idx_seq(y_train_2, y_word2idx, max_output_seq_len)

    # produce the list of the target labels in the training set
    labels_train = np.concatenate((np.ones(len(y_train_1_enc)), np.zeros(len(y_train_2_enc))))


    return (np.concatenate((np.array(x_train_enc), np.array(x_train_enc))),
            np.concatenate((np.array(y_train_1_enc), np.array(y_train_2_enc))),
            labels_train)


def load_dev_data_for_eval(data_devset, data_model_outputs_dev, vocab_size, max_input_seq_len, max_output_seq_len, delex=True):
    dataset_name = ''
    slot_sep = ''
    val_sep = ''
    val_sep_closing = False

    if '/rest_e2e/' in data_devset or '\\rest_e2e\\' in data_devset:
        x_dev, y_dev_1 = read_rest_e2e_dataset_dev(data_devset)
        y_dev_2 = read_predictions(data_model_outputs_dev)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_closing = True
    elif '/tv/' in data_devset or '\\tv\\' in data_devset:
        x_dev, y_dev_1, y_dev_2 = read_tv_dataset_dev(data_devset)
        if data_model_outputs_dev is not None:
            y_dev_2 = read_predictions(data_model_outputs_dev)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
    elif '/laptop/' in data_devset or '\\laptop\\' in data_devset:
        x_dev, y_dev_1, y_dev_2 = read_laptop_dataset_dev(data_devset)
        if data_model_outputs_dev is not None:
            y_dev_2 = read_predictions(data_model_outputs_dev)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
    else:
        raise FileNotFoundError

    # parse the utterances into lists of words
    y_dev_1 = [preprocess_utterance(y) for y in y_dev_1]
    y_dev_2 = [preprocess_utterance(y) for y in y_dev_2]
    

    # produce sequences of extracted words from the meaning representations (MRs) in the devset
    x_dev_seq = []
    for i, mr in enumerate(x_dev):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value
            
        if delex == True:
            # delexicalize the MR and the utterance
            y_dev_1[i] = delex_sample(mr_dict, y_dev_1[i], dataset=dataset_name, utterance_only=True)
            y_dev_2[i] = delex_sample(mr_dict, y_dev_2[i], dataset=dataset_name)

        # convert the dictionary to a list
        x_dev_seq.append([])
        for key, val in mr_dict.items():
            if len(val) > 0:
                x_dev_seq[i].extend([key, val])
            else:
                x_dev_seq[i].append(key)


    # load the source vocabulary
    with io.open('data/eval_vocab_source.json', 'r', encoding='utf8') as f_x_vocab:
        x_vocab = json.load(f_x_vocab)

    x_idx2word = [word[0] for word in x_vocab]
    x_idx2word.insert(0, '<PADDING>')
    x_idx2word.append('<NA>')
    x_word2idx = {word: idx for idx, word in enumerate(x_idx2word)}

    # load the target vocabulary
    with io.open('data/eval_vocab_target.json', 'r', encoding='utf8') as f_y_vocab:
        y_vocab = json.load(f_y_vocab)

    y_idx2word = [word[0] for word in y_vocab]
    y_idx2word.insert(0, '<PADDING>')
    y_idx2word.append('<NA>')
    y_word2idx = {token: idx for idx, token in enumerate(y_idx2word)}
    

    # produce sequences of indexes from the MRs in the devset
    x_dev_enc = token_seq_to_idx_seq(x_dev_seq, x_word2idx, max_input_seq_len)

    # produce sequences of indexes from the utterances in the devset
    y_dev_1_enc = token_seq_to_idx_seq(y_dev_1, y_word2idx, max_output_seq_len)

    # produce sequences of indexes from the utterances in the devset
    y_dev_2_enc = token_seq_to_idx_seq(y_dev_2, y_word2idx, max_output_seq_len)

    # produce the list of the target labels in the devset
    labels_dev = np.concatenate((np.ones(len(y_dev_1_enc)), np.zeros(len(y_dev_2_enc))))


    return (np.concatenate((np.array(x_dev_enc), np.array(x_dev_enc))),
            np.concatenate((np.array(y_dev_1_enc), np.array(y_dev_2_enc))),
            labels_dev)


def load_test_data_for_eval(data_testset, data_model_outputs_test, vocab_size, max_input_seq_len, max_output_seq_len, delex=False):
    dataset_name = ''
    slot_sep = ''
    val_sep = ''
    val_sep_closing = False

    if '/rest_e2e/' in data_testset or '\\rest_e2e\\' in data_testset:
        x_test, _ = read_rest_e2e_dataset_test(data_testset)
        y_test = read_predictions(data_model_outputs_test)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_closing = True
    elif '/tv/' in data_testset or '\\tv\\' in data_testset:
        x_test, _, y_test = read_tv_dataset_test(data_testset)
        if data_model_outputs_test is not None:
            y_test = read_predictions(data_model_outputs_test)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
    elif '/laptop/' in data_testset or '\\laptop\\' in data_testset:
        x_test, _, y_test = read_laptop_dataset_test(data_testset)
        if data_model_outputs_test is not None:
            y_test = read_predictions(data_model_outputs_test)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
    else:
        raise FileNotFoundError

    # parse the utterances into lists of words
    y_test = [preprocess_utterance(y) for y in y_test]
    #y_test_1 = [preprocess_utterance(y) for y in y_test_1]
    #y_test_2 = [preprocess_utterance(y) for y in y_test_2]
    

    # produce sequences of extracted words from the meaning representations (MRs) in the testset
    x_test_seq = []
    for i, mr in enumerate(x_test):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value

        if delex == True:
            # delexicalize the MR and the utterance
            y_test[i] = delex_sample(mr_dict, y_test[i], dataset=dataset_name)
            #y_test_1[i] = delex_sample(mr_dict, y_test_1[i], dataset=dataset_name, utterance_only=True)
            #y_test_2[i] = delex_sample(mr_dict, y_test_2[i], dataset=dataset_name)

        # convert the dictionary to a list
        x_test_seq.append([])
        for key, val in mr_dict.items():
            if len(val) > 0:
                x_test_seq[i].extend([key, val])
            else:
                x_test_seq[i].append(key)


    # load the source vocabulary
    with io.open('data/eval_vocab_source.json', 'r', encoding='utf8') as f_x_vocab:
        x_vocab = json.load(f_x_vocab)

    x_idx2word = [word[0] for word in x_vocab]
    x_idx2word.insert(0, '<PADDING>')
    x_idx2word.append('<NA>')
    x_word2idx = {word: idx for idx, word in enumerate(x_idx2word)}

    # load the target vocabulary
    with io.open('data/eval_vocab_target.json', 'r', encoding='utf8') as f_y_vocab:
        y_vocab = json.load(f_y_vocab)

    y_idx2word = [word[0] for word in y_vocab]
    y_idx2word.insert(0, '<PADDING>')
    y_idx2word.append('<NA>')
    y_word2idx = {token: idx for idx, token in enumerate(y_idx2word)}


    # produce sequences of indexes from the MRs in the test set
    x_test_enc = token_seq_to_idx_seq(x_test_seq, x_word2idx, max_input_seq_len)

    # produce sequences of indexes from the utterances in the test set
    y_test_enc = token_seq_to_idx_seq(y_test, y_word2idx, max_output_seq_len)
    #y_test_1_enc = token_seq_to_idx_seq(y_test_1, y_word2idx, max_output_seq_len)
    #y_test_2_enc = token_seq_to_idx_seq(y_test_2, y_word2idx, max_output_seq_len)

    # produce the list of the target labels in the test set
    labels_test = np.ones(len(y_test_enc))
    #labels_test = np.concatenate((np.ones(len(y_test_1_enc)), np.zeros(len(y_test_2_enc))))


    return (np.array(x_test_enc),
            np.array(y_test_enc),
            labels_test,
            x_idx2word,
            y_idx2word)

    #return (np.concatenate((np.array(x_test_enc), np.array(x_test_enc))),
    #        np.concatenate((np.array(y_test_1_enc), np.array(y_test_2_enc))),
    #        labels_test,
    #        x_idx2word,
    #        y_idx2word)


# ---- AUXILIARY FUNCTIONS ----


def init_training_data(data_trainset, data_devset):
    if 'rest_e2e' in data_trainset and 'rest_e2e' in data_devset:
        x_train, y_train = read_rest_e2e_dataset_train(data_trainset)
        x_dev, y_dev = read_rest_e2e_dataset_dev(data_devset)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_closing = True
    elif 'tv' in data_trainset and 'tv' in data_devset:
        x_train, y_train, _ = read_tv_dataset_train(data_trainset)
        x_dev, y_dev, _ = read_tv_dataset_dev(data_devset)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
        val_sep_closing = False
    elif 'laptop' in data_trainset and 'laptop' in data_devset:
        x_train, y_train, _ = read_laptop_dataset_train(data_trainset)
        x_dev, y_dev, _ = read_laptop_dataset_dev(data_devset)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
        val_sep_closing = False
    elif 'hotel' in data_trainset and 'hotel' in data_devset:
        x_train, y_train, _ = read_hotel_dataset_train(data_trainset)
        x_dev, y_dev, _ = read_hotel_dataset_dev(data_devset)
        dataset_name = 'hotel'
        slot_sep = ';'
        val_sep = '='
        val_sep_closing = False
    else:
        raise ValueError('Unexpected file name or path: {0}, {1}'.format(data_trainset, data_devset))

    return {
        'dataset_name': dataset_name,
        'data': (x_train, y_train, x_dev, y_dev),
        'separators': (slot_sep, val_sep, val_sep_closing)
    }


def init_test_data(data_testset):
    if 'rest_e2e' in data_testset:
        x_test, y_test = read_rest_e2e_dataset_test(data_testset)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_closing = True
    elif 'tv' in data_testset:
        x_test, y_test, _ = read_tv_dataset_test(data_testset)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
        val_sep_closing = False
    elif 'laptop' in data_testset:
        x_test, y_test, _ = read_laptop_dataset_test(data_testset)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
        val_sep_closing = False
    elif 'hotel' in data_testset:
        x_test, y_test, _ = read_hotel_dataset_test(data_testset)
        dataset_name = 'hotel'
        slot_sep = ';'
        val_sep = '='
        val_sep_closing = False
    else:
        raise ValueError('Unexpected file name or path: {0}'.format(data_testset))

    return {
        'dataset_name': dataset_name,
        'data': (x_test, y_test),
        'separators': (slot_sep, val_sep, val_sep_closing)
    }


def read_rest_e2e_dataset_train(data_trainset):
    # read the training data from file
    df_train = pd.read_csv(data_trainset, header=0, encoding='utf8')    # names=['mr', 'ref']
    x_train = df_train.mr.tolist()
    y_train = df_train.ref.tolist()

    return x_train, y_train


def read_rest_e2e_dataset_dev(data_devset):
    # read the development data from file
    df_dev = pd.read_csv(data_devset, header=0, encoding='utf8')        # names=['mr', 'ref']
    x_dev = df_dev.mr.tolist()
    y_dev = df_dev.ref.tolist()

    return x_dev, y_dev


def read_rest_e2e_dataset_test(data_testset):
    # read the test data from file
    df_test = pd.read_csv(data_testset, header=0, encoding='utf8')      # names=['mr', 'ref']
    x_test = df_test.iloc[:, 0].tolist()
    y_test = []
    if df_test.shape[1] > 1:
        y_test = df_test.iloc[:, 1].tolist()

    return x_test, y_test


def read_tv_dataset_train(path_to_trainset):
    with io.open(path_to_trainset, encoding='utf8') as f_trainset:
        # Skip the comment block at the beginning of the file
        f_trainset, _ = skip_comment_block(f_trainset, '#')

        # read the training data from file
        df_train = pd.read_json(f_trainset, encoding='utf8')

    x_train = df_train.iloc[:, 0].tolist()
    y_train = df_train.iloc[:, 1].tolist()
    y_train_alt = df_train.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_train):
        x_train[i] = preprocess_mr(mr, '(', ';', '=')
        
    # convert plural nouns to "[noun] -s" or "[noun] -es" form
    for i, utt in enumerate(y_train):
        y_train[i] = replace_plural_nouns(utt)
    for i, utt in enumerate(y_train_alt):
        y_train_alt[i] = replace_plural_nouns(utt)
        
    return x_train, y_train, y_train_alt


def read_tv_dataset_dev(path_to_devset):
    with io.open(path_to_devset, encoding='utf8') as f_devset:
        # Skip the comment block at the beginning of the file
        f_devset, _ = skip_comment_block(f_devset, '#')

        # read the development data from file
        df_dev = pd.read_json(f_devset, encoding='utf8')

    x_dev = df_dev.iloc[:, 0].tolist()
    y_dev = df_dev.iloc[:, 1].tolist()
    y_dev_alt = df_dev.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_dev):
        x_dev[i] = preprocess_mr(mr, '(', ';', '=')
        
    # convert plural nouns to "[noun] -s" or "[noun] -es" form
    for i, utt in enumerate(y_dev):
        y_dev[i] = replace_plural_nouns(utt)
    for i, utt in enumerate(y_dev_alt):
        y_dev_alt[i] = replace_plural_nouns(utt)

    return x_dev, y_dev, y_dev_alt


def read_tv_dataset_test(path_to_testset):
    with io.open(path_to_testset, encoding='utf8') as f_testset:
        # Skip the comment block at the beginning of the file
        f_testset, _ = skip_comment_block(f_testset, '#')

        # read the test data from file
        df_test = pd.read_json(f_testset, encoding='utf8')

    x_test = df_test.iloc[:, 0].tolist()
    y_test = df_test.iloc[:, 1].tolist()
    y_test_alt = df_test.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_test):
        x_test[i] = preprocess_mr(mr, '(', ';', '=')

    return x_test, y_test, y_test_alt


def read_laptop_dataset_train(path_to_trainset):
    with io.open(path_to_trainset, encoding='utf8') as f_trainset:
        # Skip the comment block at the beginning of the file
        f_trainset, _ = skip_comment_block(f_trainset, '#')

        # read the training data from file
        df_train = pd.read_json(f_trainset, encoding='utf8')

    x_train = df_train.iloc[:, 0].tolist()
    y_train = df_train.iloc[:, 1].tolist()
    y_train_alt = df_train.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_train):
        x_train[i] = preprocess_mr(mr, '(', ';', '=')

    return x_train, y_train, y_train_alt


def read_laptop_dataset_dev(path_to_devset):
    with io.open(path_to_devset, encoding='utf8') as f_devset:
        # Skip the comment block at the beginning of the file
        f_devset, _ = skip_comment_block(f_devset, '#')

        # read the development data from file
        df_dev = pd.read_json(f_devset, encoding='utf8')

    x_dev = df_dev.iloc[:, 0].tolist()
    y_dev = df_dev.iloc[:, 1].tolist()
    y_dev_alt = df_dev.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_dev):
        x_dev[i] = preprocess_mr(mr, '(', ';', '=')

    return x_dev, y_dev, y_dev_alt


def read_laptop_dataset_test(path_to_testset):
    with io.open(path_to_testset, encoding='utf8') as f_testset:
        # Skip the comment block at the beginning of the file
        f_testset, _ = skip_comment_block(f_testset, '#')

        # read the test data from file
        df_test = pd.read_json(f_testset, encoding='utf8')

    x_test = df_test.iloc[:, 0].tolist()
    y_test = df_test.iloc[:, 1].tolist()
    y_test_alt = df_test.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_test):
        x_test[i] = preprocess_mr(mr, '(', ';', '=')

    return x_test, y_test, y_test_alt


def read_hotel_dataset_train(path_to_trainset):
    with io.open(path_to_trainset, encoding='utf8') as f_trainset:
        # Skip the comment block at the beginning of the file
        f_trainset, _ = skip_comment_block(f_trainset, '#')

        # read the training data from file
        df_train = pd.read_json(f_trainset, encoding='utf8')

    x_train = df_train.iloc[:, 0].tolist()
    y_train = df_train.iloc[:, 1].tolist()
    y_train_alt = df_train.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_train):
        x_train[i] = preprocess_mr(mr, '(', ';', '=')

    return x_train, y_train, y_train_alt


def read_hotel_dataset_dev(path_to_devset):
    with io.open(path_to_devset, encoding='utf8') as f_devset:
        # Skip the comment block at the beginning of the file
        f_devset, _ = skip_comment_block(f_devset, '#')

        # read the development data from file
        df_dev = pd.read_json(f_devset, encoding='utf8')

    x_dev = df_dev.iloc[:, 0].tolist()
    y_dev = df_dev.iloc[:, 1].tolist()
    y_dev_alt = df_dev.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_dev):
        x_dev[i] = preprocess_mr(mr, '(', ';', '=')

    return x_dev, y_dev, y_dev_alt


def read_hotel_dataset_test(path_to_testset):
    with io.open(path_to_testset, encoding='utf8') as f_testset:
        # Skip the comment block at the beginning of the file
        f_testset, _ = skip_comment_block(f_testset, '#')

        # read the test data from file
        df_test = pd.read_json(f_testset, encoding='utf8')

    x_test = df_test.iloc[:, 0].tolist()
    y_test = df_test.iloc[:, 1].tolist()
    y_test_alt = df_test.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_test):
        x_test[i] = preprocess_mr(mr, '(', ';', '=')

    return x_test, y_test, y_test_alt


def read_predictions(path_to_predictions):
    # read the test data from file
    with io.open(path_to_predictions, encoding='utf8') as f_predictions:
        y_pred = f_predictions.readlines()

    return y_pred


def skip_comment_block(fd, comment_symbol):
    """Reads the initial lines of the file (represented by the file descriptor) corresponding to a comment block.
    All consecutive lines starting with the given symbol are considered to be part of the comment block.
    """

    comment_block = ''

    line_beg = fd.tell()
    line = fd.readline()
    while line != '':
        if not line.startswith(comment_symbol):
            fd.seek(line_beg)
            break

        comment_block += line
        line_beg = fd.tell()
        line = fd.readline()

    return fd, comment_block


def replace_plural_nouns(utt):
    stemmer = WordNetLemmatizer()

    pos_tags = nltk.pos_tag(nltk.word_tokenize(utt))
    tokens_to_replace = []
    tokens_new = []

    for token, tag in pos_tags:
        #if tag == 'NNS':
        if token in ['inches', 'watts']:
            tokens_to_replace.append(token)
            tokens_new.append(split_plural_noun(token, stemmer))
        
    for token_to_replace, token_new in zip(tokens_to_replace, tokens_new):
        utt = utt.replace(token_to_replace, token_new)

    return utt


def split_plural_noun(word, stemmer):
    stem = stemmer.lemmatize(word)
    if stem not in word or stem == word:
        return word

    suffix = word.replace(stem, '')

    return stem + ' -' + suffix


def preprocess_mr(mr, da_sep, slot_sep, val_sep):
    sep_idx = mr.find(da_sep)
    da_type = mr[:sep_idx].lstrip('?')
    slot_value_pairs = mr[sep_idx:].strip('()')

    mr_new = 'da=' + da_type
    if len(slot_value_pairs) > 0:
        mr_new += slot_sep + slot_value_pairs

    mr_modified = ''
    for slot_value in mr_new.split(slot_sep):
        slot, _, _, value_orig = parse_slot_and_value(slot_value, val_sep)
        # If the value is enclosed in apostrophes, remove them
        if value_orig.startswith('\'') and value_orig.endswith('\''):
            value_orig = value_orig[1:-1]

        mr_modified += slot + val_sep + value_orig + slot_sep

    mr_new = mr_modified[:-1]

    if da_type in ['compare', 'suggest']:
        slot_counts = {}
        mr_modified = ''
        for slot_value in mr_new.split(slot_sep):
            slot, _, _, value_orig = parse_slot_and_value(slot_value, val_sep)
            if slot in ['da', 'position']:
                mr_modified += slot
            else:
                slot_counts[slot] = slot_counts.get(slot, 0) + 1
                mr_modified += slot + str(slot_counts[slot])

            mr_modified += val_sep + value_orig + slot_sep

        mr_new = mr_modified[:-1]

    return mr_new


def preprocess_utterance(utterance):
    return word_tokenize(utterance.lower())


def parse_slot_and_value(slot_value, val_sep, val_sep_closing=False):
    sep_idx = slot_value.find(val_sep)
    if sep_idx > -1:
        # parse the slot
        slot = slot_value[:sep_idx].strip()
        # parse the value
        if val_sep_closing == True:
            value = slot_value[sep_idx + 1:-1].strip()
        else:
            value = slot_value[sep_idx + 1:].strip()
    else:
        # parse the slot
        if val_sep_closing == True:
            slot = slot_value[:-1].strip()
        else:
            slot = slot_value.strip()
        # set the value to the empty string
        value = ''

    slot_processed = slot.replace(' ', '').lower()
    value_processed = value.lower()

    return slot_processed, value_processed, slot, value


def delex_sample(mr, utterance=None, dataset=None, slots_to_delex=None, mr_only=False, input_concat=False, utterance_only=False):
    """Delexicalize a single sample (MR and the corresponding utterance).
    By default, the slots 'name', 'near' and 'food' are delexicalized (for the E2E dataset).

    All fields (E2E): name, near, area, food, customer rating, familyFriendly, eatType, priceRange
    """

    if not mr_only and utterance is None:
        raise ValueError('the \'utterance\' argument must be provided when \'mr_only\' is False.')

    if slots_to_delex is not None:
        delex_slots = slots_to_delex
    else:
        if dataset == 'rest_e2e':
            delex_slots = ['name', 'near', 'food']
        elif dataset == 'tv':
            delex_slots = ['name', 'family', 'hdmiport', 'screensize', 'price', 'audio', 'resolution', 'powerconsumption', 'color', 'count']
        elif dataset == 'laptop':
            delex_slots = ['name', 'family', 'processor', 'memory', 'drive', 'battery', 'weight', 'dimension', 'design', 'platform', 'warranty', 'count']
        elif dataset == 'hotel':
            delex_slots = ['name', 'address', 'postcode', 'area', 'near', 'phone', 'count']
        else:
            # By default, assume the dataset is 'rest_e2e'
            delex_slots = ['name', 'near', 'food']

    if not mr_only:
        utterance = ' '.join(utterance)
    mr_update = {}

    for slot, value in mr.items():
        if slot.rstrip(string.digits) in delex_slots and value not in ['dontcare', 'none', '']:
            # Assemble a placeholder token for the value
            placeholder = create_placeholder(slot, value)

            values_alt = [value]
            # Specify alternative representations of the value
            if slot == 'address':
                if 'street' in value:
                    values_alt.append(re.sub(r'\b{}\b'.format('street'), 'st', value))
                elif 'avenue' in value:
                    values_alt.append(re.sub(r'\b{}\b'.format('avenue'), 'ave', value))

            for val in values_alt:
                # Replace the value (whole-word matches only) with the placeholder
                if not mr_only:
                    utterance_delexed = re.sub(r'\b{}\b'.format(val), placeholder, utterance)
                    if utterance_delexed != utterance:
                        break
                else:
                    utterance_delexed = utterance

            # Do not replace value with a placeholder token unless there is an exact match in the utterance
            if mr_only or utterance_delexed != utterance or slot == 'name':
                mr_update[slot] = placeholder
                utterance = utterance_delexed
        else:
            if input_concat:
                mr_update[slot] = value.replace(' ', '_')

    if not utterance_only:
        for slot, new_value in mr_update.items():
            mr[slot] = new_value

    if not mr_only:
        return utterance.split()


def create_placeholder(slot, value):
    vowels = 'aeiou'

    placeholder = '<slot_'

    value = value.lower()
    if value[0] in vowels:
        placeholder += 'vow_'
    else:
        placeholder += 'con_'

    if slot == 'name':
        if value.startswith(('the ', 'a ', 'an ')):
            placeholder += 'det_'
    elif slot == 'food':
        if 'food' not in value:
            placeholder += 'cuisine_'

    placeholder += (slot + '>')

    return placeholder


def token_seq_to_idx_seq(token_seqences, token2idx, max_output_seq_len):
    # produce sequences of indexes from the utterances in the training set
    idx_sequences = np.zeros((len(token_seqences), max_output_seq_len), dtype=np.int32)       # padding implicitly present, as the index of the padding token is 0
    for i, token_seq in enumerate(token_seqences):
        for j, token in enumerate(token_seq):
            # truncate long utterances
            if j >= max_output_seq_len:
                break

            # represent each token with the corresponding index
            if token in token2idx:
                idx_sequences[i][j] = token2idx[token]
            else:
                idx_sequences[i][j] = token2idx['<NA>']

    return idx_sequences


# ---- SCRIPTS ----

def count_unique_mrs():
    """Counts unique MRs in the datasets and prints the statistics. (Requires the initial comment blocks in
    the TV and Laptop data files to be manually removed first.)
    """

    print('Unique MRs (E2E NLG):')

    df = pd.read_csv(os.path.join(config.E2E_DATA_DIR, 'trainset_e2e.csv'), header=0, encoding='utf8')
    print('train:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_csv(os.path.join(config.E2E_DATA_DIR, 'devset_e2e.csv'), header=0, encoding='utf8')
    print('valid:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_csv(os.path.join(config.E2E_DATA_DIR, 'testset_e2e.csv'), header=0, encoding='utf8')
    print('test:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))


    print('\nUnique MRs (TV):')

    df = pd.read_json(os.path.join(config.TV_DATA_DIR, 'train.json'), encoding='utf8')
    print('train:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_json(os.path.join(config.TV_DATA_DIR, 'valid.json'), encoding='utf8')
    print('valid:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_json(os.path.join(config.TV_DATA_DIR, 'test.json'), encoding='utf8')
    print('test:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))


    print('\nUnique MRs (Laptop):')

    df = pd.read_json(os.path.join(config.LAPTOP_DATA_DIR, 'train.json'), encoding='utf8')
    print('train:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_json(os.path.join(config.LAPTOP_DATA_DIR, 'valid.json'), encoding='utf8')
    print('valid:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_json(os.path.join(config.LAPTOP_DATA_DIR, 'test.json'), encoding='utf8')
    print('test:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))


    print('\nUnique MRs (Hotel):')

    df = pd.read_json(os.path.join(config.HOTEL_DATA_DIR, 'train.json'), encoding='utf8')
    print('train:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_json(os.path.join(config.HOTEL_DATA_DIR, 'valid.json'), encoding='utf8')
    print('valid:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_json(os.path.join(config.HOTEL_DATA_DIR, 'test.json'), encoding='utf8')
    print('test:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))


def verify_slot_order(dataset, filename):
    """Verifies whether the slot order in all MRs corresponds to the desired order.
    """

    slots_ordered = ['name', 'eattype', 'food', 'pricerange', 'customerrating', 'area', 'familyfriendly', 'near']
    mrs_dicts = []

    # Read in the data
    data_cont = init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, _, _, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value_orig

        mrs_dicts.append(mr_dict)

    for mr_dict in mrs_dicts:
        slots = list(mr_dict.keys())
        cur_idx = 0

        for slot in slots:
            if slot in slots_ordered:
                slot_idx = slots.index(slot)
                rightmost_idx = slots_ordered.index(slot)

                if slot_idx <= rightmost_idx and rightmost_idx >= cur_idx:
                    cur_idx = rightmost_idx
                else:
                    print('TEST FAILED: {0} has index {1} in the MR, but the order requires index {2}.'.format(
                        slot, slot_idx, slots_ordered.index(slot)))


def filter_samples_by_da_type_json(dataset, filename, das_to_keep):
    """Create a new JSON data file by filtering only those samples in the given dataset that contain an MR
    with one of the desired DA types.
    """

    if not filename.lower().endswith('.json'):
        raise ValueError('Unexpected file type. Please provide a JSON file as input.')

    data_filtered = []

    with io.open(os.path.join(config.DATA_DIR, dataset, filename), encoding='utf8') as f_dataset:
        # Skip and store the comment at the beginning of the file
        f_dataset, comment_block = skip_comment_block(f_dataset, '#')

        # Read the dataset from file
        data = json.load(f_dataset, encoding='utf8')

    # Append the opening parenthesis to the DA names, so as to avoid matching DAs whose names have these as prefixes
    das_to_keep = tuple(da + '(' for da in das_to_keep)

    # Filter MRs with the desired DA types only
    for sample in data:
        mr = sample[0]
        if mr.startswith(das_to_keep):
            data_filtered.append(sample)

    # Save the filtered dataset to a new file
    filename_out = ''.join(filename.split('.')[:-1]) + '_filtered.json'
    with io.open(os.path.join(config.DATA_DIR, dataset, filename_out), 'w', encoding='utf8') as f_dataset_filtered:
        f_dataset_filtered.write(comment_block)
        json.dump(data_filtered, f_dataset_filtered, indent=4, ensure_ascii=False)


def filter_samples_by_slot_count_csv(dataset, filename, min_count=None, max_count=None, eliminate_position_slot=True):
    """Create a new CSV data file by filtering only those samples in the given dataset that contain an MR
    with the number of slots in the desired range.
    """

    if not filename.lower().endswith('.csv'):
        raise ValueError('Unexpected file type. Please provide a CSV file as input.')

    data_filtered = []

    # Read in the data
    data_cont = init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    for mr, utt in zip(mrs, utterances):
        mr_dict = OrderedDict()
        cur_min_count = min_count or 0
        cur_max_count = max_count or 20

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            _, _, slot_orig, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot_orig] = value_orig

        if 'da' in mr_dict:
            cur_min_count += 1
            cur_max_count += 1
        if 'position' in mr_dict:
            if eliminate_position_slot:
                if mr_dict['position'] == 'inner':
                    continue
                elif mr_dict['position'] == 'outer':
                    mr = mr.replace(', position[outer]', '')
            cur_min_count += 1
            cur_max_count += 1

        if min_count is not None and len(mr_dict) < cur_min_count or \
                max_count is not None and len(mr_dict) > cur_max_count:
            continue

        data_filtered.append([mr, utt])

    # Save the filtered dataset to a new file
    filename_out = ''.join(filename.split('.')[:-1])
    if min_count is not None:
        filename_out += '_min{}'.format(min_count)
    if max_count is not None:
        filename_out += '_max{}'.format(max_count)
    filename_out += '_slots.csv'

    pd.DataFrame(data_filtered).to_csv(os.path.join(config.DATA_DIR, dataset, filename_out),
                                       header=['mr', 'ref'],
                                       index=False,
                                       encoding='utf8')


def filter_samples_by_slot_count_json(dataset, filename, min_count=None, max_count=None, eliminate_position_slot=True):
    """Create a new JSON data file by filtering only those samples in the given dataset that contain an MR
    with the number of slots in the desired range.
    """

    if not filename.lower().endswith('.json'):
        raise ValueError('Unexpected file type. Please provide a JSON file as input.')

    data_filtered = []

    with io.open(os.path.join(config.DATA_DIR, dataset, filename), encoding='utf8') as f_dataset:
        # Skip and store the comment at the beginning of the file
        _, comment_block = skip_comment_block(f_dataset, '#')

    # Read in the data
    data_cont = init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    for mr, utt in zip(mrs, utterances):
        mr_dict = OrderedDict()
        cur_min_count = min_count or 0
        cur_max_count = max_count or 20

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            _, _, slot_orig, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot_orig] = value_orig

        if 'da' in mr_dict:
            cur_min_count += 1
            cur_max_count += 1
        if 'position' in mr_dict:
            if eliminate_position_slot:
                if mr_dict['position'] == 'inner':
                    continue
                elif mr_dict['position'] == 'outer':
                    mr = mr.replace(', position[outer]', '')
            cur_min_count += 1
            cur_max_count += 1

        if min_count is not None and len(mr_dict) < cur_min_count or \
                max_count is not None and len(mr_dict) > cur_max_count:
            continue

        data_filtered.append([mr, utt, utt])

    # Save the filtered dataset to a new file
    filename_out = ''.join(filename.split('.')[:-1])
    if min_count is not None:
        filename_out += '_min{}'.format(min_count)
    if max_count is not None:
        filename_out += '_max{}'.format(max_count)
    filename_out += '_slots.json'

    with io.open(os.path.join(config.DATA_DIR, dataset, filename_out), 'w', encoding='utf8') as f_dataset_filtered:
        f_dataset_filtered.write(comment_block)
        json.dump(data_filtered, f_dataset_filtered, indent=4, ensure_ascii=False)


def get_vocab_overlap(dataset1, filename_train1, filename_dev1, dataset2, filename_train2, filename_dev2):
    data_trainset1 = os.path.join(config.DATA_DIR, dataset1, filename_train1)
    data_devset1 = os.path.join(config.DATA_DIR, dataset1, filename_dev1)
    data_trainset2 = os.path.join(config.DATA_DIR, dataset2, filename_train2)
    data_devset2 = os.path.join(config.DATA_DIR, dataset2, filename_dev2)

    dataset1 = load_training_data(data_trainset1, data_devset1)
    dataset2 = load_training_data(data_trainset2, data_devset2)

    vocab1 = get_vocabulary(dataset1)
    vocab2 = get_vocabulary(dataset2)

    common_vocab = vocab1.intersection(vocab2)

    print('Size of vocab 1:', len(vocab1))
    print('Size of vocab 2:', len(vocab2))
    print('Number of common words:', len(common_vocab))

    print('Common words:')
    print(common_vocab)


def pool_slot_values(dataset, filenames):
    """Gathers all possible values for each slot type in the dataset.
    """

    # slots_to_pool = ['eattype', 'pricerange', 'customerrating', 'familyfriendly']
    slots_to_pool = None
    slot_poss_values = {}

    # Read in the data
    if len(filenames) == 1:
        data_cont = init_test_data(os.path.join(config.DATA_DIR, dataset, filenames[0]))
        mrs, utterances = data_cont['data']
    else:
        data_cont = init_training_data(os.path.join(config.DATA_DIR, dataset, filenames[0]),
                                       os.path.join(config.DATA_DIR, dataset, filenames[1]))
        x_train, y_train, x_dev, y_dev = data_cont['data']
        mrs, utterances = (x_train + x_dev), (y_train + y_dev)

    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, _, _, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value_orig

        # For each slot gather all possible values
        for slot, value in mr_dict.items():
            slot = slot.rstrip(string.digits)
            if slots_to_pool is None or slot in slots_to_pool:
                if slot not in slot_poss_values:
                    slot_poss_values[slot] = set()
                slot_poss_values[slot].add(value)

    # Convert the value sets to lists (and make thus the dictionary serializable into JSON)
    for slot in slot_poss_values.keys():
        slot_poss_values[slot] = sorted(list(slot_poss_values[slot]))

    # Store the dictionary to a file
    with io.open(os.path.join(config.DATA_DIR, dataset, 'slot_values.json'), 'w', encoding='utf8') as f_slot_values:
        json.dump(slot_poss_values, f_slot_values, indent=4, sort_keys=True, ensure_ascii=False)


def generate_joint_vocab():
    """Generates a joint vocabulary for multiple datasets.
    """

    data_trainset = os.path.join(config.HOTEL_DATA_DIR, 'train.json')
    data_devset = os.path.join(config.HOTEL_DATA_DIR, 'valid.json')
    data_hotel = load_training_data(data_trainset, data_devset)

    data_trainset = os.path.join(config.LAPTOP_DATA_DIR, 'train.json')
    data_devset = os.path.join(config.LAPTOP_DATA_DIR, 'valid.json')
    data_laptop = load_training_data(data_trainset, data_devset)

    data_trainset = os.path.join(config.TV_DATA_DIR, 'train.json')
    data_devset = os.path.join(config.TV_DATA_DIR, 'valid.json')
    data_tv = load_training_data(data_trainset, data_devset)

    data_trainset = os.path.join(config.E2E_DATA_DIR, 'trainset_e2e_utt_split.csv')
    data_devset = os.path.join(config.E2E_DATA_DIR, 'devset_e2e.csv')
    data_rest = load_training_data(data_trainset, data_devset)

    generate_vocab_file(np.concatenate((data_rest, data_tv, data_laptop, data_hotel)),
                        vocab_filename='vocab.lang_gen.tokens')


# ---- MAIN ----

def main():
    # count_unique_mrs()

    # verify_slot_order('rest_e2e', 'trainset_e2e_utt_split.csv')

    # das_to_keep = ['inform']
    # filter_samples_by_da_type_json('tv', 'train.json', das_to_keep)
    # filter_samples_by_da_type_json('tv', 'valid.json', das_to_keep)
    # filter_samples_by_da_type_json('tv', 'test.json', das_to_keep)

    # filter_samples_by_slot_count_csv('rest_e2e', 'testset_e2e.csv', min_count=3, max_count=4)
    filter_samples_by_slot_count_json('hotel', 'test_filtered.json', min_count=3, max_count=4)

    # get_vocab_overlap('rest_e2e', 'trainset_e2e.csv', 'devset_e2e.csv',
    #                   'hotel', 'train.json', 'valid.json')
    # get_vocab_overlap('laptop', 'train.json', 'valid.json',
    #                   'tv', 'train.json', 'valid.json')

    # pool_slot_values('rest_e2e', ['trainset_e2e.csv', 'devset_e2e.csv'])
    # pool_slot_values('tv', ['train.json', 'valid.json'])
    # pool_slot_values('laptop', ['train.json', 'valid.json'])
    # pool_slot_values('hotel', ['train.json', 'valid.json'])

    # generate_joint_vocab()

    # ----------

    # x_test, y_test = read_laptop_dataset_test('data/tv/test.json')
    # print(x_test)
    # print()
    # print(y_test)
    # print()
    # print(len(x_test), len(y_test))

    # ----------

    # if len(y_test) > 0:
    #    with io.open('data/predictions_baseline.txt', 'w', encoding='utf8') as f_y_test:
    #        for line in y_test:
    #            f_y_test.write(line + '\n')

    # Produce a file from the predictions in the TV/Laptop dataset format by replacing the baseline utterances (in the 3rd column)
    # with io.open('eval/predictions-tv/predictions_ensemble_2way_2.txt', 'r', encoding='utf8') as f_predictions:
    #     with io.open('data/tv/test.json', encoding='utf8') as f_testset:
    #         # Skip the comment block at the beginning of the file
    #         f_testset, _ = skip_comment_block(f_testset, '#')
    #
    #         # read the test data from file
    #         df = pd.read_json(f_testset, encoding='utf8')
    #
    #     df.iloc[:, 2] = f_predictions.readlines()
    #     df.to_json('data/tv/test_pred.json', orient='values')

    # Produce a file from the predictions in the TV/Laptop dataset format by replacing the baseline utterances (in the 3rd column)
    # with io.open('eval/predictions-laptop/predictions_ensemble_2way_1.txt', 'r', encoding='utf8') as f_predictions:
    #     with io.open('data/laptop/test.json', encoding='utf8') as f_testset:
    #         # Skip the comment block at the beginning of the file
    #         f_testset, _ = skip_comment_block(f_testset, '#')
    #
    #         # read the test data from file
    #         df = pd.read_json(f_testset, encoding='utf8')
    #
    #     df.iloc[:, 2] = f_predictions.readlines()
    #     df.to_json('data/laptop/test_pred.json', orient='values')


if __name__ == '__main__':
    main()
