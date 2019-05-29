import os
import re
import pandas as pd
from nltk import sent_tokenize

import config
from scripts.generate_mrs import mr_to_string


class UtteranceCleaner:
    def __init__(self, file_in, da=None):
        # Read in the CSV file with slot values in separate columns
        self.file_in = file_in
        self.df_in = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8', engine='python')
        self.da = da

        self.mr_dicts = self.df_in.drop(columns=['utterance']).to_dict(orient='record')
        self.mr_dicts = [{k: v for k, v in mr_dict.items() if pd.notnull(v)} for mr_dict in self.mr_dicts]
        self.utterances = list(self.df_in['utterance'])

    def normalize(self):
        utterances_fixed = []
        change_cnt = 0

        for i, mr_dict, utt in zip(range(1, len(self.mr_dicts) + 1), self.mr_dicts, self.utterances):
            # Back up the original utterance
            utt_orig = utt

            # Substitute (multiple) white spaces with a single space, removing any leading and trailing spaces
            utt = ' '.join(utt.split())

            # Remove any empty spaces right before punctuation
            utt = re.sub(r'\s+\.$', '.', utt)
            utt = re.sub(r'\s+\?$', '?', utt)
            utt = re.sub(r'\s+\?$', '!', utt)
            utt = re.sub(r'\s+,', ',', utt)

            # Perform normalizations and capitalizations on the utterance
            utt = self.normalize_has_multiplayer(utt)
            utt = self.normalize_platforms(utt)
            utt = self.normalize_player_perspective(utt)
            utt = self.capitalize_boolean_slots(utt)
            utt = self.capitalize_categorical_slots(utt, mr_dict)

            # Capitalize sentence beginnings
            utt_capitalized = ' '.join([sent[0].upper() + sent[1:] for sent in sent_tokenize(utt)])

            utterances_fixed.append(utt_capitalized)

            # List modified utterances
            if utt != utt_orig:
                change_cnt += 1
                print('#' + str(i))
                print('ORIG.:\t' + utt_orig)
                print('CLEAN:\t' + utt_capitalized)
                if utt_capitalized != utt:
                    print('[Capitalization fixed.]')
                print()

        print('Number of changed utterances:', change_cnt)

        # Convert the MRs to their string representation and align them with the updated utterances
        df_out = pd.DataFrame()
        df_out['mr'] = [mr_to_string(mr_dict, self.da) for mr_dict in self.mr_dicts]
        df_out['ref'] = utterances_fixed

        # Save the MRs with the updated utterances to a new CSV file
        file_out = os.path.splitext(self.file_in)[0] + ' FINAL.csv'
        df_out.to_csv(file_out, index=False, encoding='utf8')

    def normalize_has_multiplayer(self, utt):
        value_map = {
            'multiplayer': ['multi player', 'multi-player', 'multiplayer'],
            'single-player': ['non multiplayer', 'non-multiplayer', 'nonmultiplayer', 'one-player',
                              'single player', 'singleplayer']
        }

        return self.__replace_values_using_map(utt, value_map)

    def normalize_platforms(self, utt):
        value_map = {
            'PC': ['pc'],
            'PlayStation': ['play station 3', 'play station 4', 'play station', 'playstation 3', 'playstation 4',
                            'ps 3', 'ps 4', 'playstation', 'ps3', 'ps4', 'ps'],
            'Xbox': ['xbox 360', 'xbox one', 'xbox360', 'xboxone', 'xbox'],
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

    def capitalize_boolean_slots(self, utt):
        value_map = {
            'Steam': ['steam'],
            'Mac': ['mac', 'macs'],
            'Linux': ['linux']
        }

        return self.__replace_values_using_map(utt, value_map)

    def capitalize_categorical_slots(self, utt, mr_dict):
        categorical_slots = {'name', 'exp_release_date', 'developer'}

        for slot, val in mr_dict.items():
            if slot in categorical_slots:
                # Replace a mixed-cased value with the given value (i.e., with correct capitalization)
                utt = self.__replace_values_using_map(utt, {val: [val]})

        return utt

    def __replace_values_using_map(self, utt, value_map):
        for target, mutations in value_map.items():
            for mutation in mutations:
                utt = re.sub(r'\b' + re.escape(mutation) + r'\b', target, utt, flags=re.IGNORECASE)

        return utt


def main():
    file_in = os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation',
                           'video_games_processed_results_inform.csv')

    cleaner = UtteranceCleaner(file_in, da='inform')

    cleaner.normalize()


if __name__ == '__main__':
    main()
