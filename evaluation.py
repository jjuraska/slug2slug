import argparse
import sys
import os
import pandas as pd
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, GRU, Dense
from keras.layers import concatenate
from keras.layers.wrappers import Bidirectional
from keras.callbacks import ModelCheckpoint

import data_loader


# ---- GLOBAL PARAMETERS ----

vocab_size = 10000                      # maximum vocabulary size of the DAs
max_mr_seq_len = 30                     # number of words the DAs should be truncated/padded to
max_utt_seq_len = 50                    # number of words the DAs should be truncated/padded to
delex = True                           # should delexicalize the samples


def main():
    # ---- PARSE ARGUMENTS ----
    
    parser = argparse.ArgumentParser(description='Perform a specific task (e.g. training, testing, prediction) with the defined model.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', nargs='+', help='takes as arguments the path to the trainset and (optionally) the path to the devset')
    group.add_argument('--test', nargs=3, help='takes as arguments the path to the test set, the path to the model outputs, and the path to the model')
    group.add_argument('--predict', nargs=3, help='takes as argument the path to the test set, the path to the model outputs, and the path to the model')

    args = parser.parse_args()

    if args.train is not None:
        if not os.path.isfile(args.train[0]) or not os.path.isfile(args.train[1]):
            print('Error: invalid file path.')
        else:
            if len(args.train) == 2:
                train(args.train[0], args.train[1])
            elif len(args.train) == 4:
                train(args.train[0], args.train[1], args.train[2], args.train[3])
            else:
                print('Error: expected 2 or 4 arguments.')
    elif args.test is not None:
        if not os.path.isfile(args.test[0]) or not os.path.isfile(args.test[1]):
            print('Error: invalid file path.')
        else:
            test(args.test[0], args.test[1], args.test[2], predict_only=False)
    elif args.predict is not None:
        if not os.path.isfile(args.predict[0]) or not os.path.isfile(args.predict[1]):
            print('Error: invalid file path.')
        else:
            test(args.predict[0], args.predict[1], args.predict[2], predict_only=True)
    else:
        print('Usage:\n')
        print('main.py')
        print('\t--train path_to_trainset [path_to_devset]')
        print('\t--test path_to_testset path_to_model_outputs path_to_model')
        print('\t--predict path_to_testset path_to_model_outputs path_to_model')


def train(data_trainset, data_model_outputs_train, data_devset=None, data_model_outputs_dev=None):
    # ---- PARAMETERS ----

    embedding_size = 300                    # dimension of the word embedding vectors
    rnn_depth = 2                           # number of RNN layers
    rnn_layer_size = 200                    # number of neurons in a single RNN layer
    num_epochs = 10                          # number of training epochs


    # ---- LOAD THE DATA ----

    print('Loading training data...', end=' ')
    sys.stdout.flush()

    mr_train, utt_train, labels_train = data_loader.load_training_data_for_eval(data_trainset,
                                                                                data_model_outputs_train,
                                                                                vocab_size,
                                                                                max_mr_seq_len,
                                                                                max_utt_seq_len,
                                                                                delex=delex)

    if data_devset is not None:
        mr_dev, utt_dev, labels_dev = data_loader.load_dev_data_for_eval(data_devset,
                                                                         data_model_outputs_dev,
                                                                         vocab_size,
                                                                         max_mr_seq_len,
                                                                         max_utt_seq_len,
                                                                         delex=delex)

    # DEBUG PRINT
    #print('---- Input overview ----')
    #print('mr_train.shape =', mr_train.shape)
    #print('utt_train.shape =', utt_train.shape)
    #print('labels_train.shape =', labels_train.shape)
    #print('----')
    
    print('DONE')


    # ---- BUILD THE MODEL ----

    print('\nBuilding the classification model...')

    # input layers
    input_mr = Input(shape=(max_mr_seq_len,), dtype='int32', name='input_mr')
    input_utt = Input(shape=(max_utt_seq_len,), dtype='int32', name='input_utt')

    # embedding layers
    embedded_mr = Embedding(vocab_size,
                            embedding_size,
                            input_length=max_mr_seq_len)(input_mr)

    embedded_utt = Embedding(vocab_size,
                             embedding_size,
                             input_length=max_utt_seq_len)(input_utt)

    # RNN encoder for MRs
    #for d in range(rnn_depth - 1):
    #    model.add(Bidirectional(GRU(units=rnn_layer_size,
    #                                 dropout=0.2,
    #                                 recurrent_dropout=0.2,
    #                                 return_sequences=True)))
    
    encoded_mr = Bidirectional(GRU(units=rnn_layer_size,
                                   dropout=0.2,
                                   recurrent_dropout=0.2,
                                   return_sequences=False))(embedded_mr)

    # RNN encoder for utterances
    #for d in range(rnn_depth - 1):
    #    model.add(Bidirectional(GRU(units=rnn_layer_size,
    #                                 dropout=0.2,
    #                                 recurrent_dropout=0.2,
    #                                 return_sequences=True)))
    
    encoded_utt = Bidirectional(GRU(units=rnn_layer_size,
                                    dropout=0.2,
                                    recurrent_dropout=0.2,
                                    return_sequences=False))(embedded_utt)

    # merge layers
    x = concatenate([encoded_mr, encoded_utt], axis=-1)

    # fully-connected layers
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)

    # fully-connected output layer
    prediction = Dense(1, activation='sigmoid')(x)

    # create a model from the graph
    model = Model(inputs=[input_mr, input_utt], outputs=[prediction])


    # ---- COMPILE THE MODEL ----

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()

    # define checkpoint callback (a model checkpoint will be created at the end of each epoch)
    filepath = 'model/model-checkpoint-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]

    
    # ---- TRAIN THE MODEL ----

    print('Training...')
    sys.stdout.flush()

    if data_devset is not None:
        history = model.fit([mr_train, utt_train],
                            labels_train,
                            validation_data=([mr_dev, utt_dev], labels_dev),
                            batch_size=64,
                            epochs=num_epochs,
                            callbacks=callback_list)
    else:
        history = model.fit([mr_train, utt_train],
                            labels_train,
                            batch_size=64,
                            epochs=num_epochs,
                            callbacks=callback_list)

    print('DONE')
    
    
def test(data_testset, data_model_outputs, path_to_model, predict_only=True):
    # ---- LOAD THE DATA ----

    vocab_source_file = 'data/eval_vocab_source.json'
    vocab_target_file = 'data/eval_vocab_target.json'
    evaluations_file = 'predictions/predictions.csv'

    print('Loading test data...', end=' ')
    sys.stdout.flush()

    if not os.path.isfile(vocab_source_file) or not os.path.isfile(vocab_target_file):
        raise FileNotFoundError('Vocabulary files missing.')
        
    mr_test, utt_test, labels_test, mr_idx2word, utt_idx2word = data_loader.load_test_data_for_eval(data_testset,
                                                                                                    data_model_outputs,
                                                                                                    vocab_size,
                                                                                                    max_mr_seq_len,
                                                                                                    max_utt_seq_len,
                                                                                                    delex=delex)

    # DEBUG PRINT
    #print('---- Input overview ----')
    #print('mr_pred.shape =', mr_pred.shape)
    #print('utt_pred.shape =', utt_pred.shape)
    #print('labels_pred.shape =', labels_pred.shape)
    #print('----')
    
    print('DONE')
    

    # ---- LOAD THE MODEL ----

    print('\nLoading the model...')

    # load the model from a checkpoint
    model = load_model(path_to_model)
    model.summary()


    # ---- TEST THE MODEL ----

    if not predict_only:
        print('\nEvaluating...')

        loss, acc = model.evaluate([mr_test, utt_test],
                                    labels_test)

        print()
        print('-> Test loss:', loss)
        print('-> Test accuracy:', acc)


    # ---- PERFORM INFERENCE ----

    print('\nPredicting...', end=' ')

    results = []
    prediction_distr = model.__predict_single_input([mr_test, utt_test])
    predictions = np.array(prediction_distr).flatten()
    
    for i, class_predicted in enumerate(predictions):
        mr_pred_words = ' '.join([mr_idx2word[idx] for idx in mr_test[i] if idx > 0])
        utt_pred_words = ' '.join([utt_idx2word[idx] for idx in utt_test[i] if idx > 0])
        results.append([mr_pred_words, utt_pred_words, class_predicted, labels_test[i]])

    # save the results to a CSV file along with the corresponding DAs and reference classes
    df = pd.DataFrame(np.asarray(results))
    df.to_csv(evaluations_file, header=['MR', 'utterance', 'prediction', 'ref'], index=False)
    
    print('DONE')


if __name__ == "__main__":
    main()
