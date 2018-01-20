import sys
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
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import itertools


class SetEncoder(json.JSONEncoder):
   def default(self, obj):
      if isinstance(obj, set):
         return list(obj)
      return json.JSONEncoder.default(self, obj)


def load_training_data(data_trainset, data_devset, input_concat=False):
    dataset_name = ''
    slot_sep = ''
    val_sep = ''
    val_sep_closing = False

    if '/rest_e2e/' in data_trainset and '/rest_e2e/' in data_devset or \
            '\\rest_e2e\\' in data_trainset and '\\rest_e2e\\' in data_devset:
        x_train, y_train, x_dev, y_dev = read_rest_e2e_dataset_train(data_trainset, data_devset)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_closing = True
    elif '/tv/' in data_trainset and '/tv/' in data_devset or \
            '\\tv\\' in data_trainset and '\\tv\\' in data_devset:
        x_train, y_train, x_dev, y_dev = read_tv_dataset_train(data_trainset, data_devset)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
    elif '/laptop/' in data_trainset and '/laptop/' in data_devset or \
            '\\laptop\\' in data_trainset and '\\laptop\\' in data_devset:
        x_train, y_train, _, x_dev, y_dev, _ = read_laptop_dataset_train(data_trainset, data_devset)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
    else:
        raise FileNotFoundError

    # parse the utterances into lists of words
    y_train = [preprocess_utterance(y) for y in y_train]
    y_dev = [preprocess_utterance(y) for y in y_dev]

    slot_poss_values = {}

    # produce sequences of extracted words from the meaning representations (MRs) in the trainset
    x_train_seq = []
    for i, mr in enumerate(x_train):
        try:
            mr_dict = OrderedDict()
            for slot_value in mr.split(slot_sep):
                slot, value = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
                mr_dict[slot.lower()] = value.lower()

                # collect all possible values for each slot
                key_clean = slot.rstrip(string.digits)
                if key_clean not in slot_poss_values:
                    slot_poss_values[key_clean] = set([value.lower()])
                else:
                    slot_poss_values[key_clean].add(value.lower())
        except:
            print(str(mr))
            print(str(y_train[i]))
            exit()

        # delexicalize the MR and the utterance
        y_train[i] = delex_sample(mr_dict, y_train[i], input_concat=input_concat)

        # convert the dictionary to a list
        x_train_seq.append([])
        for key, val in mr_dict.items():
            if len(val) > 0:
                x_train_seq[i].extend([key, val])
            else:
                x_train_seq[i].append(key)

        if input_concat:
            # append a sequence-end token to be paired up with seq2seq's sequence-end token when concatenating
            x_train_seq[i].append('&stop&')


    # produce sequences of extracted words from the meaning representations (MRs) in the devset
    x_dev_seq = []
    for i, mr in enumerate(x_dev):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot.lower()] = value.lower()

            # collect all possible values for each slot
            key_clean = slot.rstrip(string.digits)
            if key_clean not in slot_poss_values:
                slot_poss_values[key_clean] = set([value.lower()])
            else:
                slot_poss_values[key_clean].add(value.lower())

        # delexicalize the MR and the utterance
        y_dev[i] = delex_sample(mr_dict, y_dev[i], input_concat=input_concat)

        # convert the dictionary to a list
        x_dev_seq.append([])
        for key, val in mr_dict.items():
            if len(val) > 0:
                x_dev_seq[i].extend([key, val])
            else:
                x_dev_seq[i].append(key)

        if input_concat:
            # append a sequence-end token to be paired up with seq2seq's sequence-end token when concatenating
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

    with io.open('data/slot_values.json', 'w', encoding='utf8') as f_slot_values:
        json.dump(slot_poss_values, f_slot_values, cls=SetEncoder, indent=4)


def load_test_data(data_testset, input_concat=False):
    dataset_name = ''
    slot_sep = ''
    val_sep = ''
    val_sep_closing = False

    if '/rest_e2e/' in data_testset or '\\rest_e2e\\' in data_testset:
        x_test, y_test = read_rest_e2e_dataset_test(data_testset)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_closing = True
    elif '/tv/' in data_testset or '\\tv\\' in data_testset:
        x_test, y_test = read_tv_dataset_test(data_testset)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
    elif '/laptop/' in data_testset or '\\laptop\\' in data_testset:
        x_test, y_test, _ = read_laptop_dataset_test(data_testset)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
    else:
        raise FileNotFoundError

    slots_with_proper_nouns = ['name', 'near', 'area', 'food']
    vocab_proper_nouns = set()

    # produce sequences of extracted words from the meaning representations (MRs) in the testset
    x_test_seq = []
    x_test_dict = []
    for i, mr in enumerate(x_test):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value = parse_slot_and_value(slot_value, val_sep, val_sep_closing)

            # store proper noun values (for retrieval in postprocessing)
            if slot in slots_with_proper_nouns and len(value) > 0 and value[0].isupper():
                vocab_proper_nouns.add(value)

            mr_dict[slot.lower()] = value.lower()

        # build the MR dictionary
        x_test_dict.append(copy.deepcopy(mr_dict))

        # delexicalize the MR
        delex_sample(mr_dict, mr_only=True, input_concat=input_concat)

        # convert the dictionary to a list
        x_test_seq.append([])
        for key, val in mr_dict.items():
            if len(val) > 0:
                x_test_seq[i].extend([key, val])
            else:
                x_test_seq[i].append(key)

        if input_concat:
            # append a sequence-end token to be paired up with seq2seq's sequence-end token when concatenating
            x_test_seq[i].append('&stop&')

    with io.open('data/test_source.txt', 'w', encoding='utf8') as f_x_test:
        for line in x_test_seq:
            f_x_test.write('{}\n'.format(' '.join(line)))

    with io.open('data/test_source_dict.json', 'w', encoding='utf8') as f_x_test_dict:
        json.dump(x_test_dict, f_x_test_dict)

    # vocabulary of proper nouns to be used for capitalization in postprocessing
    with io.open('data/vocab_proper_nouns.txt', 'w', encoding='utf8') as f_vocab:
        for value in vocab_proper_nouns:
            f_vocab.write(value + '\n')

    if len(y_test) > 0:
        with io.open('data/test_target.txt', 'w', encoding='utf8') as f_y_test:
            for line in y_test:
                f_y_test.write(line + '\n')

        # reference file for calculating metrics for test predictions
        with io.open('metrics/test_references.txt', 'w', encoding='utf8') as f_y_test:
            for i, line in enumerate(y_test):
                if i > 0 and x_test[i] != x_test[i - 1]:
                    f_y_test.write('\n')
                f_y_test.write(line + '\n')


def load_training_data_for_eval(data_trainset, data_devset, vocab_size, max_input_seq_len, max_output_seq_len):
    dataset_name = ''
    slot_sep = ''
    val_sep = ''
    val_sep_closing = False

    if '/rest_e2e/' in data_trainset and '/rest_e2e/' in data_devset or \
            '\\rest_e2e\\' in data_trainset and '\\rest_e2e\\' in data_devset:
        x_train, y_train, x_dev, y_dev = read_rest_e2e_dataset_train(data_trainset, data_devset)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_closing = True
    elif '/tv/' in data_trainset and '/tv/' in data_devset or \
            '\\tv\\' in data_trainset and '\\tv\\' in data_devset:
        x_train, y_train, x_dev, y_dev = read_tv_dataset_train(data_trainset, data_devset)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
    elif '/laptop/' in data_trainset and '/laptop/' in data_devset or \
            '\\laptop\\' in data_trainset and '\\laptop\\' in data_devset:
        x_train, y_train, y_train_alt, x_dev, y_dev, y_dev_alt = read_laptop_dataset_train(data_trainset, data_devset)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
    else:
        raise FileNotFoundError

    # parse the utterances into lists of words
    y_train = [preprocess_utterance(y) for y in y_train]
    y_train_alt = [preprocess_utterance(y) for y in y_train_alt]
    

    # produce sequences of extracted words from the meaning representations (MRs) in the trainset
    x_train_seq = []
    for i, mr in enumerate(x_train):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot.lower()] = value.lower()

        # delexicalize the MR and the utterance
        y_train[i] = delex_sample(mr_dict, y_train[i], utterance_only=True)
        y_train_alt[i] = delex_sample(mr_dict, y_train_alt[i])

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

    #with io.open('data/eval_vocab_source_ordered.json', 'w', encoding='utf8') as f_x_vocab:
    #    json.dump(OrderedDict(x_vocab), f_x_vocab, indent=4, ensure_ascii=False)

    # create target vocabulary
    if os.path.isfile('data/eval_vocab_target.json'):
        with io.open('data/eval_vocab_target.json', 'r', encoding='utf8') as f_y_vocab:
            y_vocab = json.load(f_y_vocab)
    else:
        y_distr = FreqDist([y_token for y in y_train for y_token in y] + [y_token for y in y_train_alt for y_token in y])
        y_vocab = y_distr.most_common(min(len(y_distr), vocab_size - 2))        # cap the vocabulary size
        with io.open('data/eval_vocab_target.json', 'w', encoding='utf8') as f_y_vocab:
            json.dump(y_vocab, f_y_vocab, ensure_ascii=False)

    y_idx2word = [word[0] for word in y_vocab]
    y_idx2word.insert(0, '<PADDING>')
    y_idx2word.append('<NA>')
    y_word2idx = {token: idx for idx, token in enumerate(y_idx2word)}

    #with io.open('data/eval_vocab_target_ordered.json', 'w', encoding='utf8') as f_y_vocab:
    #    json.dump(OrderedDict(y_vocab), f_y_vocab, indent=4, ensure_ascii=False)


    # produce sequences of indexes from the MRs in the training set
    x_train_enc = np.zeros((len(x_train_seq), max_input_seq_len), dtype=np.int32)       # padding implicitly present, as the index of the padding token is 0
    for i, x in enumerate(x_train_seq):
        for j, token in enumerate(x):
            # truncate long MRs
            if j >= max_input_seq_len:
                break

            # represent each token with the corresponding index
            if token in x_word2idx:
                x_train_enc[i][j] = x_word2idx[token]
            else:
                x_train_enc[i][j] = x_word2idx['<NA>']

    # produce sequences of indexes from the utterances in the training set
    y_train_enc = np.zeros((len(y_train), max_output_seq_len), dtype=np.int32)       # padding implicitly present, as the index of the padding token is 0
    for i, y in enumerate(y_train):
        for j, token in enumerate(y):
            # truncate long utterances
            if j >= max_output_seq_len:
                break

            # represent each token with the corresponding index
            if token in y_word2idx:
                y_train_enc[i][j] = y_word2idx[token]
            else:
                y_train_enc[i][j] = y_word2idx['<NA>']

    # produce sequences of indexes from the utterances in the training set
    y_train_alt_enc = np.zeros((len(y_train_alt), max_output_seq_len), dtype=np.int32)       # padding implicitly present, as the index of the padding token is 0
    for i, y in enumerate(y_train_alt):
        for j, token in enumerate(y):
            # truncate long utterances
            if j >= max_output_seq_len:
                break

            # represent each token with the corresponding index
            if token in y_word2idx:
                y_train_alt_enc[i][j] = y_word2idx[token]
            else:
                y_train_alt_enc[i][j] = y_word2idx['<NA>']

    # produce the list of the target labels in the training set
    labels = np.concatenate((np.ones(len(y_train_enc)), np.zeros(len(y_train_alt_enc))))


    return (np.concatenate((np.array(x_train_enc), np.array(x_train_enc))),
            np.concatenate((np.array(y_train_enc), np.array(y_train_alt_enc))),
            labels)


def load_test_data_for_eval(data_testset, vocab_size, max_input_seq_len, max_output_seq_len):
    dataset_name = ''
    slot_sep = ''
    val_sep = ''
    val_sep_closing = False

    if '/rest_e2e/' in data_testset or '\\rest_e2e\\' in data_testset:
        x_test, y_test = read_rest_e2e_dataset_test(data_testset)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_closing = True
    elif '/tv/' in data_testset or '\\tv\\' in data_testset:
        x_test, y_test = read_tv_dataset_test(data_testset)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
    elif '/laptop/' in data_testset or '\\laptop\\' in data_testset:
        x_test, y_test, y_test_alt = read_laptop_dataset_test(data_testset)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
    else:
        raise FileNotFoundError

    # parse the utterances into lists of words
    y_test = [preprocess_utterance(y) for y in y_test]
    y_test_alt = [preprocess_utterance(y) for y in y_test_alt]
    

    # produce sequences of extracted words from the meaning representations (MRs) in the testset
    x_test_seq = []
    for i, mr in enumerate(x_test):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value = parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot.lower()] = value.lower()

        # delexicalize the MR and the utterance
        y_test[i] = delex_sample(mr_dict, y_test[i], utterance_only=True)
        y_test_alt[i] = delex_sample(mr_dict, y_test_alt[i])

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


    # produce sequences of indexes from the MRs in the training set
    x_test_enc = np.zeros((len(x_test_seq), max_input_seq_len), dtype=np.int32)       # padding implicitly present, as the index of the padding token is 0
    for i, x in enumerate(x_test_seq):
        for j, token in enumerate(x):
            # truncate long MRs
            if j >= max_input_seq_len:
                break

            # represent each token with the corresponding index
            if token in x_word2idx:
                x_test_enc[i][j] = x_word2idx[token]
            else:
                x_test_enc[i][j] = x_word2idx['<NA>']

    # produce sequences of indexes from the utterances in the training set
    y_test_enc = np.zeros((len(y_test), max_output_seq_len), dtype=np.int32)       # padding implicitly present, as the index of the padding token is 0
    for i, y in enumerate(y_test):
        for j, token in enumerate(y):
            # truncate long utterances
            if j >= max_output_seq_len:
                break

            # represent each token with the corresponding index
            if token in y_word2idx:
                y_test_enc[i][j] = y_word2idx[token]
            else:
                y_test_enc[i][j] = y_word2idx['<NA>']

    # produce sequences of indexes from the utterances in the training set
    y_test_alt_enc = np.zeros((len(y_test_alt), max_output_seq_len), dtype=np.int32)       # padding implicitly present, as the index of the padding token is 0
    for i, y in enumerate(y_test_alt):
        for j, token in enumerate(y):
            # truncate long utterances
            if j >= max_output_seq_len:
                break

            # represent each token with the corresponding index
            if token in y_word2idx:
                y_test_alt_enc[i][j] = y_word2idx[token]
            else:
                y_test_alt_enc[i][j] = y_word2idx['<NA>']

    # produce the list of the target labels in the training set
    labels = np.concatenate((np.ones(len(y_test_enc)), np.zeros(len(y_test_alt_enc))))


    return (np.concatenate((np.array(x_test_enc), np.array(x_test_enc))),
            np.concatenate((np.array(y_test_enc), np.array(y_test_alt_enc))),
            labels)


# ---- AUXILIARY FUNCTIONS ----

def read_rest_e2e_dataset_train(data_trainset, data_devset):
    # read the training data from file
    df_train = pd.read_csv(data_trainset, header=0, encoding='utf8')    # names=['mr', 'ref']
    x_train = df_train.mr.tolist()
    y_train = df_train.ref.tolist()

    # read the development data from file
    df_dev = pd.read_csv(data_devset, header=0, encoding='utf8')        # names=['mr', 'ref']
    x_dev = df_dev.mr.tolist()
    y_dev = df_dev.ref.tolist()

    return x_train, y_train, x_dev, y_dev


def read_rest_e2e_dataset_test(data_testset):
    # read the test data from file
    df_test = pd.read_csv(data_testset, header=0, encoding='utf8')  # names=['mr', 'ref']
    x_test = df_test.iloc[:, 0].tolist()
    y_test = []
    if df_test.shape[1] > 1:
        y_test = df_test.iloc[:, 1].tolist()

    return x_test, y_test


def read_tv_dataset_train(path_to_trainset, path_to_devset):
    with io.open(path_to_trainset, encoding='utf8') as f_trainset:
        # remove the comment at the beginning of the file
        for i in range(5):
            f_trainset.readline()

        # read the training data from file
        df_train = pd.read_json(f_trainset, encoding='utf8')

    x_train = df_train.iloc[:, 0].tolist()
    y_train = df_train.iloc[:, 1].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_train):
        x_train[i] = preprocess_mr(mr, '(', ';', '=')
        
    # convert plural nouns to "[noun] -s" or "[noun] -es" form
    stemmer = WordNetLemmatizer()
    for i, utt in enumerate(y_train):
        y_train[i] = replace_plural_nouns(utt)

        
    with io.open(path_to_devset, encoding='utf8') as f_devset:
        # remove the comment at the beginning of the file
        for i in range(5):
            f_devset.readline()

        # read the development data from file
        df_dev = pd.read_json(f_devset, encoding='utf8')

    x_dev = df_dev.iloc[:, 0].tolist()
    y_dev = df_dev.iloc[:, 1].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_dev):
        x_dev[i] = preprocess_mr(mr, '(', ';', '=')
        
    # convert plural nouns to "[noun] -s" or "[noun] -es" form
    stemmer = WordNetLemmatizer()
    for i, utt in enumerate(y_dev):
        y_dev[i] = replace_plural_nouns(utt)

    return x_train, y_train, x_dev, y_dev


def read_tv_dataset_test(path_to_testset):
    with io.open(path_to_testset, encoding='utf8') as f_testset:
        # remove the comment at the beginning of the file
        for i in range(5):
            f_testset.readline()

        # read the test data from file
        df_test = pd.read_json(f_testset, encoding='utf8')

    x_test = df_test.iloc[:, 0].tolist()
    y_test = df_test.iloc[:, 1].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_test):
        x_test[i] = preprocess_mr(mr, '(', ';', '=')

    return x_test, y_test


def read_laptop_dataset_train(path_to_trainset, path_to_devset):
    with io.open(path_to_trainset, encoding='utf8') as f_trainset:
        # remove the comment at the beginning of the file
        for i in range(5):
            f_trainset.readline()

        # read the training data from file
        df_train = pd.read_json(f_trainset, encoding='utf8')

    x_train = df_train.iloc[:, 0].tolist()
    y_train = df_train.iloc[:, 1].tolist()
    y_train_alt = df_train.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_train):
        x_train[i] = preprocess_mr(mr, '(', ';', '=')


    with io.open(path_to_devset, encoding='utf8') as f_devset:
        # remove the comment at the beginning of the file
        for i in range(5):
            f_devset.readline()

        # read the development data from file
        df_dev = pd.read_json(f_devset, encoding='utf8')

    x_dev = df_dev.iloc[:, 0].tolist()
    y_dev = df_dev.iloc[:, 1].tolist()
    y_dev_alt = df_dev.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_dev):
        x_dev[i] = preprocess_mr(mr, '(', ';', '=')

    return x_train, y_train, y_train_alt, x_dev, y_dev, y_dev_alt


def read_laptop_dataset_test(path_to_testset):
    with io.open(path_to_testset, encoding='utf8') as f_testset:
        # remove the comment at the beginning of the file
        for i in range(5):
            f_testset.readline()

        # read the test data from file
        df_test = pd.read_json(f_testset, encoding='utf8')

    x_test = df_test.iloc[:, 0].tolist()
    y_test = df_test.iloc[:, 1].tolist()
    y_test_alt = df_test.iloc[:, 2].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_test):
        x_test[i] = preprocess_mr(mr, '(', ';', '=')

    return x_test, y_test, y_test_alt


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

    if da_type in ['compare', 'suggest']:
        slot_counts = {}
        mr_modified = ''
        for slot_value in mr_new.split(slot_sep):
            slot, value = parse_slot_and_value(slot_value, val_sep)
            if slot in ['da', 'position']:
                mr_modified += slot
            else:
                slot_counts[slot] = slot_counts.get(slot, 0) + 1
                mr_modified += slot + str(slot_counts[slot])

            mr_modified += val_sep + value + slot_sep

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
                    
    slot = slot.replace(' ', '_')

    return (slot, value)


def delex_sample(mr, utterance=None, slots_to_delex=None, mr_only=False, input_concat=False, utterance_only=False):
    '''
    Delexicalize a single sample (MR and the corresponding utterance).
    By default, the slots 'name' and 'near' are delexicalized.
    All fields: name, near, area, food, customer rating, familyFriendly, eatType, priceRange
    '''

    vowels = 'aeiou'

    if not mr_only and utterance == None:
        raise ValueError('the \'utterance\' argument must be provided when \'mr_only\' is False.')
        return None

    if slots_to_delex is not None:
        delex_slots = slots_to_delex
    else:
        delex_slots = ['name', 'near', 'food',
                       'family', 'hdmiport', 'screensize', 'price', 'audio', 'resolution', 'powerconsumption', 'color', 'count',
                       'processor', 'memory', 'drive', 'battery', 'weight', 'dimension', 'design', 'platform', 'warranty']

    if not mr_only:
        utterance = ' '.join(utterance)
    mr_update = {}

    for slot, value in mr.items():
        if slot.rstrip(string.digits) in delex_slots and value not in ['dontcare', 'none', '']:
            placeholder = '&slot_'
            if value[0].lower() in vowels:
                placeholder += 'vow_'
            else:
                placeholder += 'con_'

            if slot == 'name':
                if value.lower().startswith(('the ', 'a ', 'an ')):
                    placeholder += 'det_'
            elif slot == 'food':
                if 'food' not in value.lower():
                    placeholder += 'cuisine_'

            placeholder += (slot + '&')

            utterance_delexed = utterance
            if not mr_only:
                utterance_delexed = re.sub(r'\b{}\b'.format(value), placeholder, utterance)     # replace whole-word matches only

            # don't replce value with a placeholder token unless there is an exact match in the utterance
            if mr_only or utterance_delexed != utterance or (slot == 'name'):
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


def count_unique_mrs():
    print('Unique MRs (E2E NLG):')

    df = pd.read_csv('data/rest_e2e/trainset_e2e.csv', header=0, encoding='utf8')
    print('train:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_csv('data/rest_e2e/devset_e2e.csv', header=0, encoding='utf8')
    print('valid:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_csv('data/rest_e2e/testset_e2e.csv', header=0, encoding='utf8')
    print('test:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))


    print('\nUnique MRs (Laptop):')

    df = pd.read_json('data/laptop/train.json', encoding='utf8')
    print('train:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_json('data/laptop/valid.json', encoding='utf8')
    print('valid:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_json('data/laptop/test.json', encoding='utf8')
    print('test:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))


    print('\nUnique MRs (TV):')

    df = pd.read_json('data/tv/train.json', encoding='utf8')
    print('train:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_json('data/tv/valid.json', encoding='utf8')
    print('valid:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))

    df = pd.read_json('data/tv/test.json', encoding='utf8')
    print('test:\t', len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))


# ---- MAIN ----

if __name__ == '__main__':
    #count_unique_mrs()

    #x_test, y_test = read_laptop_dataset_test('data/tv/test.json')

    #print(x_test)
    #print()
    #print(y_test)
    #print()
    #print(len(x_test), len(y_test))

    #if len(y_test) > 0:
    #    with io.open('data/predictions_baseline.txt', 'w', encoding='utf8') as f_y_test:
    #        for line in y_test:
    #            f_y_test.write(line + '\n')


    # produce a file from the predictions in the TV/Laptop dataset format by replacing the baseline utterances (in the 3rd column)
    with io.open('eval/predictions-tv/predictions_ensemble_2way_2.txt', 'r', encoding='utf8') as f_predictions:
        with io.open('data/tv/test.json', encoding='utf8') as f_testset:
            # remove the comment at the beginning of the file
            for i in range(5):
                f_testset.readline()

            # read the test data from file
            df = pd.read_json(f_testset, encoding='utf8')

        df.iloc[:, 2] = f_predictions.readlines()
        df.to_json('data/tv/test_pred.json', orient='values')


    # produce a file from the predictions in the TV/Laptop dataset format by replacing the baseline utterances (in the 3rd column)
    #with io.open('eval/predictions-laptop/predictions_ensemble_2way_1.txt', 'r', encoding='utf8') as f_predictions:
    #    with io.open('data/laptop/test.json', encoding='utf8') as f_testset:
    #        # remove the comment at the beginning of the file
    #        for i in range(5):
    #            f_testset.readline()

    #        # read the test data from file
    #        df = pd.read_json(f_testset, encoding='utf8')

    #    df.iloc[:, 2] = f_predictions.readlines()
    #    df.to_json('data/laptop/test_pred.json', orient='values')
