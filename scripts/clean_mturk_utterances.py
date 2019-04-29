import os
import re
import pandas as pd

import config


class UtteranceCleaner:

    def __init__(self, file_in):
        # Read in the CSV file with slot values in separate columns
        self.df_input = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8')
        self.utterances = list(self.df_input['utterance'])

    def normalize(self):
        utterances_normalized = []
        change_cnt = 0

        for utt in self.utterances:
            # Back up the original utterance
            utt_orig = utt

            # Perform normalizations on the utterance
            utt = self.normalize_has_multiplayer(utt)
            utt = self.normalize_platforms(utt)
            utt = self.normalize_player_perspective(utt)

            utterances_normalized.append(utt)

            # List modified utterances
            if utt != utt_orig:
                change_cnt += 1
                print('ORIG.:\t' + utt_orig)
                print('NORM.:\t' + utt + '\n')

        print('Number of changed utterances:', change_cnt)

    def normalize_has_multiplayer(self, utt):
        value_map = {
            'multiplayer': ['multi player', 'multi-player'],
            'single-player': ['non multiplayer', 'non-multiplayer', 'nonmultiplayer']
        }

        return self.__replace_values_using_map(utt, value_map)

    def normalize_platforms(self, utt):
        value_map = {
            'PC': ['pc'],
            'PlayStation': ['play station 3', 'play station 4', 'play station', 'playstation 3', 'playstation 4',
                            'ps 3', 'ps 4', 'ps3', 'ps4', 'ps'],
            'Xbox': ['xbox 360', 'xbox one', 'xbox360', 'xboxone'],
            'Nintendo Switch': ['nintendo switch'],
            'Nintendo': ['nintendo']
        }

        return self.__replace_values_using_map(utt, value_map)

    def normalize_player_perspective(self, utt):
        value_map = {
            'first person': ['first-person', '1st person', '1st-person'],
            'third person': ['third-person', '3rd person', '3rd-person'],
            'bird view': ['bird-view'],
            'side view': ['side-view']
        }

        return self.__replace_values_using_map(utt, value_map)

    def __replace_values_using_map(self, utt, value_map):
        for target, mutations in value_map.items():
            for mutation in mutations:
                utt = re.sub(r'\b' + re.escape(mutation) + r'\b', target, utt, flags=re.IGNORECASE)

        return utt


def main():
    file_in = os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation',
                           'video_games_processed_results_round1_give_opinion (4 slots).csv')

    cleaner = UtteranceCleaner(file_in)

    cleaner.normalize()


if __name__ == '__main__':
    main()
