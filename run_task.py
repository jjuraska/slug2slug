import argparse
import sys
import os
import glob
import io
import json
import random
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
    group.add_argument('--test_from_beams', nargs=2, help='takes as arguments the path to the testset and the beams directory')
    group.add_argument('--test_all', nargs=1, help='takes as argument the path to the testset')
    group.add_argument('--predict', nargs=1, help='takes as argument the path to the testset')

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
    elif args.test_from_beams is not None:
        if not os.path.isfile(args.test_from_beams[0]):
            print('Error: invalid file path.')
        else:
            test_from_beams(args.test_from_beams[0], args.test_from_beams[1], predict_only=False)
    elif args.test_all is not None:
        if not os.path.isfile(args.test_all[0]):
            print('Error: invalid file path.')
        else:
            test_all(args.test_all[0])
    elif args.predict is not None:
        if not os.path.isfile(args.predict[0]):
            print('Error: invalid file path.')
        else:
            test(args.predict[0], predict_only=True)
    else:
        print('Usage:\n')
        print('run_task.py')
        print('\t--train [path_to_trainset] [path_to_devset]')
        print('\t--test [path_to_testset]')
        print('\t--test_from_beams [path_to_testset] [path_to_beams]')
        print('\t--test_all [path_to_testset]')
        print('\t--predict [path_to_testset]')


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
    test_reference_file = os.path.join(config.METRICS_DIR, 'test_references.txt')

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
        beams = postprocessing.rerank_beams(beams)

    print('DONE')
    print('Evaluating...')
    sys.stdout.flush()

    with io.open(test_source_file, 'r', encoding='utf8') as f_test_source, \
            io.open(predictions_final_file, 'w', encoding='utf8') as f_predictions_final:

        mrs = json.load(f_test_source, object_pairs_hook=OrderedDict)
        predictions = [prediction_beams[0][0] for prediction_beams in beams]
        predictions_final = postprocessing.finalize_utterances(predictions, mrs)

        for prediction in predictions_final:
            f_predictions_final.write(prediction + '\n')

        if not predict_only:
            # Create a file with a single prediction for each group of the same MRs
            if 'rest_e2e' in data_testset:
                test_mrs, _ = data_loader.read_rest_e2e_dataset_test(data_testset)
            elif 'tv' in data_testset:
                test_mrs, _, _ = data_loader.read_tv_dataset_test(data_testset)
            elif 'laptop' in data_testset:
                test_mrs, _, _ = data_loader.read_laptop_dataset_test(data_testset)
            elif 'hotel' in data_testset:
                test_mrs, _, _ = data_loader.read_hotel_dataset_test(data_testset)
            elif 'video_game' in data_testset:
                test_mrs, _ = data_loader.read_video_game_dataset_test(data_testset)
            else:
                raise FileNotFoundError

            with io.open(predictions_reduced_file, 'w', encoding='utf8') as f_predictions_reduced:
                for i in range(len(test_mrs)):
                    if i == 0 or test_mrs[i] != test_mrs[i - 1]:
                        f_predictions_reduced.write(predictions_final[i] + '\n')

    if not predict_only:
        # Depending on the OS, the tensor2tensor BLEU script might require a different way of executing
        if sys.executable is not None:
            bleu_script = 'python ' + os.path.join(os.path.dirname(sys.executable), 't2t-bleu')
        else:
            bleu_script = 't2t-bleu'

        metrics_script = 'python ' + os.path.join(config.METRICS_DIR, 'measure_scores.py')

        # Run the tensor2tensor internal BLEU script
        os.system(bleu_script +
                  ' --translation=' + predictions_final_file +
                  ' --reference=' + test_target_file)

        # Run the metrics script provided by the E2E NLG Challenge
        os.system(metrics_script + ' ' + test_reference_file + ' ' + predictions_reduced_file)

    print('DONE')


def test_from_beams(data_testset, beams_dir, predict_only=True, sample_best=False):
    test_source_file = os.path.join(config.DATA_DIR, 'test_source_dict.json')
    test_target_file = os.path.join(config.DATA_DIR, 'test_target.txt')
    predictions_final_file = os.path.join(config.PREDICTIONS_DIR, 'predictions_final.txt')
    predictions_reduced_file = os.path.join(config.METRICS_DIR, 'predictions_reduced.txt')
    test_reference_file = os.path.join(config.METRICS_DIR, 'test_references.txt')

    print('Loading test data...', end=' ')
    sys.stdout.flush()

    # Load and preprocess the test data
    data_loader.load_test_data(data_testset)

    print('DONE')
    print('Extracting beams...')
    sys.stdout.flush()

    # Read all beam files in the given beams folder
    beam_files = glob.glob(os.path.join(beams_dir, '*.txt'))

    print('-> Beam files found:')
    print('\n'.join(beam_files))

    # Combine all beam files into a single DataFrame
    df_beams = pd.concat((pd.read_csv(f, sep='\t', header=None, encoding='utf8') for f in beam_files), axis=1, ignore_index=True)
    assert len(df_beams.columns) > 1

    # Combine beams and their corresponding scores into tuples
    beams = []
    for i in range(0, len(df_beams.columns), 2):
        beams.append(list(zip(df_beams.iloc[:, i], df_beams.iloc[:, i+1])))

    # Transpose the list of beams so as to have all beams of a single sample per line
    beams = list(map(list, zip(*beams)))

    print('DONE')
    print('Reranking...')
    sys.stdout.flush()

    # Score the slot alignment in the beams, and rerank the beams accordingly
    if sample_best:
        beams = postprocessing.rerank_beams(beams, keep_n=10, keep_least_errors_only=True)
    else:
        beams = postprocessing.rerank_beams(beams, keep_n=10)

    print('DONE')
    print('Evaluating...')
    sys.stdout.flush()

    with io.open(test_source_file, 'r', encoding='utf8') as f_test_source, \
            io.open(predictions_final_file, 'w', encoding='utf8') as f_predictions_final:

        mrs = json.load(f_test_source, object_pairs_hook=OrderedDict)

        if sample_best:
            predictions = [random.choice(prediction_beams)[0] for prediction_beams in beams]
        else:
            predictions = [prediction_beams[0][0] for prediction_beams in beams]

        # Post-process the generated utterances
        predictions_final = postprocessing.finalize_utterances(predictions, mrs)

        for prediction in predictions_final:
            f_predictions_final.write(prediction + '\n')

        if not predict_only:
            # Create a file with a single prediction for each group of the same MRs
            if 'rest_e2e' in data_testset:
                test_mrs, _ = data_loader.read_rest_e2e_dataset_test(data_testset)
            elif 'tv' in data_testset:
                test_mrs, _, _ = data_loader.read_tv_dataset_test(data_testset)
            elif 'laptop' in data_testset:
                test_mrs, _, _ = data_loader.read_laptop_dataset_test(data_testset)
            elif 'hotel' in data_testset:
                test_mrs, _, _ = data_loader.read_hotel_dataset_test(data_testset)
            elif 'video_game' in data_testset:
                test_mrs, _ = data_loader.read_video_game_dataset_test(data_testset)
            else:
                raise FileNotFoundError

            with io.open(predictions_reduced_file, 'w', encoding='utf8') as f_predictions_reduced:
                for i in range(len(test_mrs)):
                    if i == 0 or test_mrs[i] != test_mrs[i - 1]:
                        f_predictions_reduced.write(predictions_final[i] + '\n')

    if not predict_only:
        # Depending on the OS, the tensor2tensor BLEU script might require a different way of executing
        if sys.executable is not None:
            bleu_script = 'python ' + os.path.join(os.path.dirname(sys.executable), 't2t-bleu')
        else:
            bleu_script = 't2t-bleu'

        metrics_script = 'python ' + os.path.join(config.METRICS_DIR, 'measure_scores.py')

        # Run the tensor2tensor internal BLEU script
        os.system(bleu_script +
                  ' --translation=' + predictions_final_file +
                  ' --reference=' + test_target_file)

        # Run the metrics script provided by the E2E NLG Challenge
        os.system(metrics_script + ' ' + test_reference_file + ' ' + predictions_reduced_file)

    print('DONE')


# In order to make it work on Windows, do the following modification in the tensor2tensor files:
# -> utils/bleu_hook.py: in function _read_stepfiles_list(), change "*-[0-9]*" to "*"
def test_all(data_testset, reranking=True):
    test_source_file = os.path.join(config.DATA_DIR, 'test_source_dict.json')
    test_target_file = os.path.join(config.DATA_DIR, 'test_target.txt')

    # Prepare the output folder
    if not os.path.exists(config.PREDICTIONS_BATCH_LEX_DIR):
        os.makedirs(config.PREDICTIONS_BATCH_LEX_DIR)

    print('Loading test data...', end=' ')
    sys.stdout.flush()

    # Load and preprocess the test data
    data_loader.load_test_data(data_testset)

    print('DONE')
    print('Predicting...')
    sys.stdout.flush()

    # Run inference for the test samples using each checkpoint of the model
    os.system('bash ' + os.path.join(config.T2T_DIR, 't2t_test_all_script.sh'))

    print('DONE')
    print('Evaluating...')
    sys.stdout.flush()

    # Relexicalize all prediction files
    for predictions_file in glob.glob(os.path.join(config.PREDICTIONS_BATCH_DIR, '*')):
        predictions_final_file = os.path.join(config.PREDICTIONS_BATCH_LEX_DIR, os.path.basename(predictions_file))

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

        # Score the slot alignment in the beams, and rerank the beams accordingly
        if reranking and beams_present:
            beams = postprocessing.rerank_beams(beams)

        # Postprocess the generated utterances and save them to a new file
        with io.open(test_source_file, 'r', encoding='utf8') as f_test_source, \
                io.open(predictions_final_file, 'w', encoding='utf8') as f_predictions_final:

            mrs = json.load(f_test_source, object_pairs_hook=OrderedDict)
            predictions = [prediction_beams[0][0] for prediction_beams in beams]
            predictions_final = postprocessing.finalize_utterances(predictions, mrs)

            for prediction in predictions_final:
                f_predictions_final.write(prediction + '\n')

    # Depending on the OS, the tensor2tensor BLEU script might require a different way of executing
    if sys.executable is not None:
        bleu_script = 'python ' + os.path.join(os.path.dirname(sys.executable), 't2t-bleu')
    else:
        bleu_script = 't2t-bleu'

    # Run the tensor2tensor internal BLEU script
    os.system(bleu_script +
              ' --translations_dir=' + config.PREDICTIONS_BATCH_LEX_DIR +
              ' --reference=' + test_target_file +
              ' --event_dir=' + config.PREDICTIONS_BATCH_EVENT_DIR)

    print('DONE')


if __name__ == '__main__':
    sys.exit(int(main() or 0))
