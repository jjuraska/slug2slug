import argparse
import sys
import os
import io
import json
import pandas as pd
from collections import OrderedDict

import config
import data_loader
import postprocessing


def main():
    parser = argparse.ArgumentParser(
        description='Perform a specific task (e.g. training, testing, prediction) with the defined model.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', nargs=2, help='takes as arguments the paths to the trainset and the devset')
    group.add_argument('--test', nargs=1, help='takes as argument the path to the testset')
    group.add_argument('--predict', nargs=1, help='takes as argument the path to the testset')
    group.add_argument('--beam_dump', nargs=1, help='takes as argument the path to the testset')

    args = parser.parse_args()

    if args.train is not None:
        if not os.path.isfile(args.train[0]) or not os.path.isfile(args.train[1]):
            print('Error: invalid file path.')
        else:
            train(args.train[0], args.train[1])
    elif args.test is not None:
        if not os.path.isfile(args.test[0]):
            print('Error: invalid file path.')
        else:
            test(args.test[0], predict_only=False)
    elif args.predict is not None:
        if not os.path.isfile(args.predict[0]):
            print('Error: invalid file path.')
        else:
            test(args.predict[0], predict_only=True)
    elif args.beam_dump is not None:
        if not os.path.isfile(args.beam_dump[0]):
            print('Error: invalid file path.')
        else:
            postprocessing.get_utterances_from_beam(args.beam_dump[0])
    else:
        print('Usage:\n')
        print('main.py')
        print('\t--train [path_to_trainset] [path_to_devset]')
        print('\t--test [path_to_testset]')
        print('\t--predict [path_to_testset]')
        print('\t--beam_dump [path_to_beams]')


def train(data_trainset, data_devset):
    print('Loading training data...', end=' ')
    sys.stdout.flush()

    # Load and preprocess the training and validation data
    data_loader.load_training_data(data_trainset, data_devset, generate_vocab=True)

    # Generate the data files in the format required for T2T
    os.system('bash ' + os.path.join(config.T2T_DIR, 't2t_datagen_script.sh'))

    print('DONE')
    print('Training...')
    sys.stdout.flush()

    # Run the model training
    os.system('bash ' + os.path.join(config.T2T_DIR, 't2t_train_script.sh'))

    print('DONE')


def test(data_testset, predict_only=True, reranking=True):
    test_source_file = os.path.join(config.DATA_DIR, 'test_source_dict.json')
    test_target_file = os.path.join(config.DATA_DIR, 'test_target.txt')
    predictions_file = os.path.join(config.PREDICTIONS_DIR, 'predictions.txt')
    predictions_final_file = os.path.join(config.PREDICTIONS_DIR, 'predictions_final.txt')
    predictions_reduced_file = os.path.join(config.METRICS_DIR, 'predictions_reduced.txt')

    print('Loading test data...', end=' ')
    sys.stdout.flush()

    # Load and preprocess the test data
    data_loader.load_test_data(data_testset)

    print('DONE')
    print('Predicting...')
    sys.stdout.flush()

    # TODO: set DECODE_FILE and PREDICTION_FILE environment variables from here instead of the shell script

    # Run inference for the test samples
    os.system('bash ' + os.path.join(config.T2T_DIR, 't2t_test_script.sh'))

    print('DONE')
    print('Extracting beams...')
    sys.stdout.flush()

    # Read in the beams and their log-probs as produced by the T2T beam search
    df_predictions = pd.read_csv(predictions_file, sep='\t', header=None, encoding='utf8')
    beams_present = len(df_predictions.columns) > 1

    if beams_present:
        # Combine beams and their corresponding scores into tuples
        beams = []
        for i in range(0, len(df_predictions.columns), 2):
            beams.append(list(zip(df_predictions.iloc[:, i], df_predictions.iloc[:, i+1])))

        # Transpose the list of beams so as to have all beams of a single sample per line
        beams = list(map(list, zip(*beams)))
    else:
        beams = [[(beam,)] for beam in df_predictions.iloc[:, 0].tolist()]

    print('DONE')
    print('Reranking...')
    sys.stdout.flush()

    # Score the slot alignment in the beams, and rerank the beams accordingly
    if reranking and beams_present:
        beams = postprocessing.align_beams_t2t(beams)

    print('DONE')
    print('Evaluating...')
    sys.stdout.flush()

    with io.open(predictions_file, 'r', encoding='utf8') as f_predictions, \
            io.open(test_source_file, 'r', encoding='utf8') as f_test_source, \
            io.open(predictions_final_file, 'w', encoding='utf8') as f_predictions_final:

        mrs = json.load(f_test_source, object_pairs_hook=OrderedDict)
        predictions = [prediction_beams[0][0] for prediction_beams in beams]
        predictions_final = postprocessing.finalize_utterances(predictions, mrs)

        for prediction in predictions_final:
            f_predictions_final.write(prediction + '\n')

        if not predict_only:
            # create a file with a single prediction for each group of the same MRs
            if '/rest_e2e/' in data_testset or '\\rest_e2e\\' in data_testset:
                test_mrs, _ = data_loader.read_rest_e2e_dataset_test(data_testset)
            elif '/tv/' in data_testset or '\\tv\\' in data_testset:
                test_mrs, _ = data_loader.read_tv_dataset_test(data_testset)
            elif '/laptop/' in data_testset or '\\laptop\\' in data_testset:
                test_mrs, _ = data_loader.read_laptop_dataset_test(data_testset)
            else:
                raise FileNotFoundError

            with io.open(predictions_reduced_file, 'w', encoding='utf8') as f_predictions_reduced:
                for i in range(len(test_mrs)):
                    if i == 0 or test_mrs[i] != test_mrs[i - 1]:
                        f_predictions_reduced.write(predictions_final[i] + '\n')

    if not predict_only:
        os.system('perl ../bin/tools/multi-bleu.perl ' + test_target_file + ' < ' + predictions_final_file)

    print('DONE')


if __name__ == '__main__':
    sys.exit(int(main() or 0))
