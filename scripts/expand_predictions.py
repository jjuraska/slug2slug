"""Expand the output predictions file as a reverse process of prediction reduction due to repeating MRs.
(Assuming equivalent MRs are grouped together.)
"""

import os
import pandas as pd

import config


def main():
    predictions_reduced_file = os.path.join(config.METRICS_DIR, 'predictions_rnn_4+4_utt_split_16k_reranked.txt')
    predictions_file = os.path.join(config.PREDICTIONS_DIR, 'predictions.txt')

    with open(predictions_reduced_file, 'r', encoding='utf8') as f_predictions_reduced:
        predictions = f_predictions_reduced.read().splitlines()
        idx = -1

        # Create a file with a repeated prediction across each group of the same MRs
        data_frame_test = pd.read_csv(os.path.join(config.E2E_DATA_DIR, 'devset_e2e.csv'), header=0, encoding='utf8')
        test_mrs = data_frame_test.mr.tolist()

        with open(predictions_file, 'w', encoding='utf8') as f_predictions:
            for i in range(len(test_mrs)):
                if i == 0 or test_mrs[i] != test_mrs[i - 1]:
                    idx += 1

                f_predictions.write(predictions[idx] + '\n')


if __name__ == '__main__':
    main()
