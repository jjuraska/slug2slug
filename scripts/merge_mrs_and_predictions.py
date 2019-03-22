"""Create a new CSV file that has the dataset's MRs merged with the generated utterances (predictions).
"""

import os
import pandas as pd
import numpy as np

import config


def main():
    dataset_file = os.path.join(config.E2E_DATA_DIR, 'testset_e2e_augm_emphasis_contrast_combo_extra.csv')
    predictions_file = os.path.join(config.EVAL_DIR, 'predictions rest_e2e (emphasis+contrast)',
                                    'predictions TRANS emphasis+contrast, train all, test combo extra (30.4k iter).txt')
    out_file = os.path.splitext(predictions_file)[0] + '.csv'

    with open(predictions_file, 'r', encoding='utf8') as f_predictions:
        predictions = f_predictions.read().splitlines()

        df_test = pd.read_csv(dataset_file, header=0, encoding='utf8')
        mrs = df_test.mr.tolist()

        assert(len(mrs) == len(predictions))

        df_out = pd.DataFrame(np.vstack((mrs, predictions)).transpose(), columns=['mr', 'ref'])
        df_out.to_csv(out_file, index=False, encoding='utf8')


if __name__ == '__main__':
    main()
