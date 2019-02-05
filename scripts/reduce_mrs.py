"""Reduce the output predictions file so there is a single utterance for each unique MR.
(Assuming equivalent MRs are grouped together.)
"""

import os
import pandas as pd

import config


def main():
    input_file = os.path.join(config.VIDEO_GAME_DATA_DIR, 'test.csv')
    output_file = os.path.join(config.VIDEO_GAME_DATA_DIR, 'test_mrs_reduced.csv')

    df_in = pd.read_csv(input_file, header=0, encoding='utf8')
    mrs_unique = df_in.mr.unique().tolist()

    df_out = pd.DataFrame(mrs_unique, columns=['mr'])
    df_out.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
