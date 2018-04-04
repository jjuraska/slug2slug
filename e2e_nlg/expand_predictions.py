'''
Expand the output predictions file as a reverse process of prediction reduction due to repeating MRs.
(Assuming equivalent MRs are grouped together.)
'''

import io
import pandas as pd


predictions_reduced_file = 'metrics/predictions_rnn_4+4_utt_split_16k_reranked.txt'
predictions_file = 'predictions/predictions.txt'

with io.open(predictions_reduced_file, 'r', encoding='utf8') as f_predictions_reduced:
    predictions = f_predictions_reduced.read().splitlines()
    idx = -1

    # create a file with a single prediction for each group of the same MRs
    data_frame_test = pd.read_csv('data/rest_e2e/devset_e2e.csv', header=0, encoding='utf8')
    test_mrs = data_frame_test.mr.tolist()

    with io.open(predictions_file, 'w', encoding='utf8') as f_predictions:
        for i in range(len(test_mrs)):
            if i == 0 or test_mrs[i] != test_mrs[i - 1]:
                idx += 1

            f_predictions.write(predictions[idx] + '\n')
