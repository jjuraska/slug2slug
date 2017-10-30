import argparse
import sys
import os
import io
import json
import platform
import pandas as pd
import numpy as np

import data_loader
import postprocessing

def test():
    predictions_file = 'predictions/predictions.txt'
    predictions_final_file = 'predictions/predictions_final.txt'
    predictions_reduced_file = 'metrics/predictions_reduced.txt'
    test_source_file = 'data/test_source_dict.json'
    test_target_file = 'data/test_target.txt'
    vocab_file = 'data/vocab_proper_nouns.txt'
    data_testset = 'data/testset.csv'

    if not os.path.isfile(test_source_file) or \
            not os.path.isfile(test_target_file) or \
            not os.path.isfile(vocab_file):
        data_loader.load_test_data(data_testset, input_concat=False)

    with io.open(predictions_file, 'r', encoding='utf8') as f_predictions:
        with io.open(test_source_file, 'r', encoding='utf8') as f_test_source:
            with io.open(predictions_final_file, 'w', encoding='utf8') as f_predictions_final:
                mrs = json.load(f_test_source)
                predictions = f_predictions.read().splitlines()
                predictions_final = postprocessing.finalize_utterances(predictions, mrs)

                for prediction in predictions_final:
                    f_predictions_final.write(prediction + '\n')

                # create a file with a single prediction for each group of the same MRs
                data_frame_test = pd.read_csv('data/testset.csv', header=0, encoding='utf8')
                test_mrs = data_frame_test.mr.tolist()

                with io.open(predictions_reduced_file, 'w', encoding='utf8') as f_predictions_reduced:
                    for i in range(len(test_mrs)):
                        if i == 0 or test_mrs[i] != test_mrs[i - 1]:
                            f_predictions_reduced.write(predictions_final[i] + '\n')

    os.system('perl ../bin/tools/multi-bleu.perl ' + test_target_file + ' < ' + predictions_final_file)

    print('DONE')


def main():
    test()
if __name__ == '__main__':
    main()
