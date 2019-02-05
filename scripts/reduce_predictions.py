"""Reduce the output predictions file so there is a single utterance for each unique MR.
(Assuming equivalent MRs are grouped together.)
"""

import os
import pandas as pd

import config


def main():
    predictions_file = os.path.join(config.PREDICTIONS_DIR, 'predictions.txt')
    predictions_reduced_file = os.path.join(config.METRICS_DIR, 'predictions_reduced_NEW.txt')

    with open(predictions_file, 'r', encoding='utf8') as f_predictions:
        predictions = f_predictions.read().splitlines()

        # Create a file with a single prediction for each group of the same MRs
        data_frame_test = pd.read_csv(os.path.join(config.VIDEO_GAME_DATA_DIR, 'test.csv'), header=0, encoding='utf8')
        test_mrs = data_frame_test.mr.tolist()

        with open(predictions_reduced_file, 'w', encoding='utf8') as f_predictions_reduced:
            for i in range(len(test_mrs)):
                if i == 0 or test_mrs[i] != test_mrs[i - 1]:
                    f_predictions_reduced.write(predictions[i] + '\n')


if __name__ == '__main__':
    main()
