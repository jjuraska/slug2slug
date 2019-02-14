import os
import random
import re
import json
import pandas as pd
import numpy as np
from collections import OrderedDict

import config


NUM_VARIATIONS = 10


class MRGenerator:
    def __init__(self, domain):
        if domain == 'video_game':
            with open(os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation', 'slots.json'), 'r') as f_slots:
                slots_dict = json.load(f_slots)
        else:
            raise ValueError('Unexpected domain "' + domain + '"')

        self.slots = slots_dict['all']
        self.da_slots = slots_dict['dialogue_acts']

    def create_mrs_from_game_data(self, file_in, da=None, num_slots=None):
        mrs = []
        mr_dicts = []

        # Initialize the MR length distribution dictionary
        mr_len_distr = OrderedDict()
        for i in range(3, 9):
            mr_len_distr[str(i)] = 0

        # Initialize the slot distribution dictionary
        slot_distr = OrderedDict()
        for s in self.slots:
            slot_distr[s] = 0

        # Read in the CSV file with the video game data
        df = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8')

        for _, row in df.iterrows():
            mr_var_cnt = 0
            while mr_var_cnt < NUM_VARIATIONS:
                # Generate a random MR from the current game's data
                mr_dict = self.__get_random_slot_comb_for_da(row, da, num_slots)

                # Assemble the MR in a string form
                mr = self.__mr_to_string(mr_dict, da)

                # If such an MR variant has not been generated yet, save it
                if mr not in mrs[-NUM_VARIATIONS:]:
                    # DEBUG PRINT
                    # if len(mr_dict) > 8:
                    #     print('CHECK:', mr_dict)

                    mrs.append(mr)
                    mr_dicts.append(mr_dict)

                    mr_len_distr[str(len(mr_dict))] = mr_len_distr.get(str(len(mr_dict)), 0) + 1
                    for slot in mr_dict:
                        slot_distr[slot] += 1

                    mr_var_cnt += 1

                # The not-yet-released games typically have fewer slots, so fewer MR variants are possible for them
                if 'exp_release_date' in mr_dict and mr_var_cnt >= 4:
                    break

        # Print the MR length distribution and the slot distribution
        print('MR length distribution:\n')
        print_stats_from_dict(mr_len_distr)
        print('\nSlot distribution:\n')
        print_stats_from_dict(slot_distr)

        # Store the generated MRs to a text file
        file_out = os.path.splitext(file_in)[0] + '_mrs{0}.txt'.format(('_' + da) if da is not None else '')
        with open(file_out, 'w', encoding='utf8') as f_out:
            f_out.write('\n'.join(mrs))

        # Format the generated MRs for a HIT and store them in a CSV file
        file_out_hit = os.path.splitext(file_in)[0] + '_mrs{0}_hit.csv'.format(('_' + da) if da is not None else '')
        self.__save_csv_for_hit(mr_dicts, file_out_hit)

    def create_mrs_from_csv(self, file_in, file_out):
        """Generates MRs in a textual form from an input CSV file in which each column corresponds to one slot.
        """

        mrs = []

        # Initialize the MR length distribution dictionary
        mr_len_distr = OrderedDict()
        for i in range(3, 9):
            mr_len_distr[str(i)] = 0

        # Initialize the slot distribution dictionary
        slot_distr = OrderedDict()
        for s in self.slots:
            slot_distr[s] = 0

        # Read in the CSV file with slot values in separate columns
        df = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8')

        for _, row in df.iterrows():
            # Remove all NaN values from the row
            mr_dict = OrderedDict(row[~row.isnull()])

            # Assemble the MR in a string form
            mr = self.__mr_to_string(mr_dict)

            mrs.append(mr)

            mr_len_distr[str(len(mr_dict))] = mr_len_distr.get(str(len(mr_dict)), 0) + 1
            for slot in mr_dict:
                slot_distr[slot] += 1

        # Print the MR length distribution and the slot distribution
        print('MR length distribution:\n')
        print_stats_from_dict(mr_len_distr)
        print('\nSlot distribution:\n')
        print_stats_from_dict(slot_distr)

        # Store the generated MRs to a text file
        with open(file_out, 'w', encoding='utf8') as f_out:
            f_out.write('\n'.join(mrs))

    def __get_random_slot_comb_for_da(self, row, da='inform', num_slots=1):
        if da == 'inform':
            mr_dict = self.__get_random_slot_comb_inform(row)
        else:
            mr_dict = OrderedDict()
            mandatory_slots, excluded_slots = self.__load_slots_for_da(da)

            # Create a slot-value dict out of all non-NaN values in the row
            slot_value_dict = OrderedDict(row[~row.isnull()])

            # Initialize the MR dict with the mandatory slots for this DA
            for slot in mandatory_slots:
                if slot in slot_value_dict:
                    mr_dict[slot] = slot_value_dict.pop(slot, None)

            # Remove all slots that should not be considered for this DA
            for slot in excluded_slots:
                slot_value_dict.pop(slot, None)

            # Sample additional slots from the remaining
            num_additional_slots = num_slots - len(mandatory_slots)
            if num_additional_slots > 0:
                mr_dict.update(random.sample(slot_value_dict.items(), num_additional_slots))

            self.__adjust_mr_for_da(da, mr_dict)

        return mr_dict

    def __get_random_slot_comb_inform(self, row):
        mr_dict = OrderedDict()
        mandatory_slots, excluded_slots = self.__load_slots_for_da('inform')
        pc_op_systems = ['has_linux_release', 'has_mac_release']
        pc_platforms = ['available_on_steam'] + pc_op_systems
        linux_mac_remove_flag = False

        # Remove all NaN values from the row
        slot_value_dict = OrderedDict(row[~row.isnull()])

        if 'developer' in slot_value_dict:
            if np.random.random() < 0.5:
                del slot_value_dict['developer']
        if 'esrb' in slot_value_dict:
            if np.random.random() < 0.5:
                del slot_value_dict['esrb']
        if 'has_multiplayer' in slot_value_dict:
            if np.random.random() < 0.25:
                del slot_value_dict['has_multiplayer']

        max_slots = int(min(max(np.round(np.random.normal(loc=6.5, scale=1.5)), 3), 8))
        max_slots = min(max_slots, len(slot_value_dict))
        num_slots = len(slot_value_dict)

        # Generate a random permutation of the slot indexes
        perm_idxs = np.random.permutation(num_slots)

        # DEBUG PRINT
        # print(num_slots)
        # print(perm_idxs)

        for slot_idx in perm_idxs:
            # Stop removing slots as soon as the MR has been reduced to the desired size
            if num_slots <= max_slots:
                break

            slot, val = list(slot_value_dict.items())[slot_idx]

            # Skip slots that were already removed due to one of the dependency rules (see below)
            if slot_value_dict[slot] is None:
                continue

            # Skip the mandatory slots
            if slot in mandatory_slots:
                continue

            # Apply slot dependency rules
            if slot == 'platforms':
                pc_platform_slot_cnt = 0
                for s in pc_platforms:
                    if slot_value_dict.get(s, None) is not None:
                        pc_platform_slot_cnt += 1

                if num_slots - pc_platform_slot_cnt - 1 >= 3:
                    slot_value_dict['platforms'] = None
                    num_slots -= 1
                    for s in pc_platforms:
                        if slot_value_dict.get(s, None) is not None:
                            slot_value_dict[s] = None
                            num_slots -= 1
            elif slot in pc_op_systems:
                if num_slots - len(pc_op_systems) >= max_slots:
                    if linux_mac_remove_flag:
                        for s in pc_op_systems:
                            if s in slot_value_dict:
                                slot_value_dict[s] = None
                                num_slots -= 1
                    else:
                        linux_mac_remove_flag = True
            elif slot == 'available_on_steam':
                if np.random.random() < 0.5:
                    continue
                else:
                    slot_value_dict['available_on_steam'] = None
                    num_slots -= 1
            else:
                # Mark the current slot for removal
                slot_value_dict[slot] = None
                num_slots -= 1

        # Remove the marked slots from the MR
        for slot, val in slot_value_dict.items():
            if val:
                mr_dict[slot] = val

        return mr_dict

    def __load_slots_for_da(self, da):
        try:
            mandatory_slots = self.da_slots[da]['mandatory']
        except KeyError:
            mandatory_slots = []

        try:
            excluded_slots = self.da_slots[da]['excluded']
        except KeyError:
            excluded_slots = []

        return mandatory_slots, excluded_slots

    def __adjust_mr_for_da(self, da, mr_dict):
        if da == 'request_attribute':
            # Remove all slots' values
            for slot in mr_dict:
                mr_dict[slot] = None
        elif da == 'request_explanation':
            # Sample a single element per slot, whenever the slot's value is a list
            for slot, val in mr_dict.items():
                if isinstance(val, list):
                    mr_dict[slot] = random.sample(val, 1)
        elif da == 'request':
            # Prepend an empty "category" slot
            mr_dict['category'] = None
            mr_dict.move_to_end('category', last=False)

    def __mr_to_string(self, mr_dict, da=None):
        slot_value_pairs = []

        for slot, val in mr_dict.items():
            slot_value_pairs.append(slot + '[' + str(val) + ']')

        mr = ', '.join(slot_value_pairs)

        if da is not None:
            # Prepend the DA, and enclose the list of the MR's slot-value pairs in parentheses
            mr = da + '(' + mr + ')'

        return mr

    def __save_csv_for_hit(self, mr_dicts, filepath):
        # Add HTML formatting to the values
        for mr_dict in mr_dicts:
            for slot, val in mr_dict.items():
                if slot == 'name':
                    mr_dict[slot] = '<b>' + val + '</b>'
                elif slot == 'release_year':
                    mr_dict[slot] = 'Year: ' + '<b>' + val + '</b>'
                elif slot == 'exp_release_date':
                    mr_dict[slot] = 'Expected release date: ' + '<b>' + val + '</b>'
                elif slot == 'developer':
                    mr_dict[slot] = 'Developer: ' + '<b>' + val + '</b>'
                elif slot == 'esrb':
                    mr_dict[slot] = 'ESRB: ' + '<b>' + val + '</b>'
                elif slot == 'rating':
                    mr_dict[slot] = 'Liking/Rating: ' + '<b>' + val + '</b>'
                elif slot == 'genres':
                    mr_dict[slot] = 'Genres: ' + '<b>' + val + '</b>'
                elif slot == 'player_perspective':
                    mr_dict[slot] = 'Player perspective: ' + '<b>' + val + '</b>'
                elif slot == 'has_multiplayer':
                    mr_dict[slot] = 'Has multiplayer: ' + '<b>' + val + '</b>'
                elif slot == 'platforms':
                    mr_dict[slot] = 'Platforms: ' + '<b>' + val + '</b>'
                elif slot == 'available_on_steam':
                    mr_dict[slot] = 'Available on Steam: ' + '<b>' + val + '</b>'
                elif slot == 'has_linux_release':
                    mr_dict[slot] = 'Has a Linux release: ' + '<b>' + val + '</b>'
                elif slot == 'has_mac_release':
                    mr_dict[slot] = 'Has a Mac release: ' + '<b>' + val + '</b>'

        # Store the formatted MRs to a CSV file
        df_hit = pd.DataFrame(mr_dicts, columns=self.slots)
        df_hit.to_csv(filepath, index=False, encoding='utf8')


def shuffle_samples_csv(filepath):
    df = pd.read_csv(filepath, header=0, dtype=object, encoding='utf8')
    df = df.sample(frac=1).reset_index(drop=True)

    filepath_out = filepath.rstrip('.csv') + '_shuffled.csv'
    df.to_csv(filepath_out, index=False, encoding='utf8')


def train_test_split(dataset_filepath, misses_filepath):
    valid_ratio = 0.2
    test_ratio = 0.2

    non_train_ratio = valid_ratio + test_ratio

    trainset = []
    validset = []
    testset = []

    train_mrs_slots_only = set()

    df_dataset = pd.read_csv(dataset_filepath, header=0, encoding='utf8')
    df_misses = pd.read_csv(misses_filepath, header=0, encoding='utf8')

    num_samples = df_dataset.shape[0]
    num_refs_per_mr = 3

    # DEBUG PRINT
    # print(num_samples)
    # print(df_misses.dtypes)

    if df_misses.shape[0] != num_samples:
        raise ValueError('Input files do not have matching numbers of lines.')

    cur_row = 0
    for mr_idx in range(num_samples // num_refs_per_mr):
        mr_samples = []
        has_ref_with_miss = False
        for ref_idx in range(num_refs_per_mr):
            mr_samples.append(tuple(df_dataset.iloc[cur_row, :].values))
            if df_misses.iat[cur_row, 2] > 0:
                has_ref_with_miss = True
            cur_row += 1

        # DEBUG PRINT
        # print(cur_row, '->', has_ref_with_miss)
        # print(mr_samples)

        # Partition the samples into train/valid/test
        if not has_ref_with_miss and random.random() < non_train_ratio and \
                remove_vals_from_slots(mr_samples[0][0], selected_slots_only=True) not in train_mrs_slots_only:
            if random.random() < (valid_ratio / non_train_ratio):
                validset.extend(mr_samples)
            else:
                testset.extend(mr_samples)
        else:
            # All samples with missing/hallucinated information go to trainset
            trainset.extend(mr_samples)
            train_mrs_slots_only.add(remove_vals_from_slots(mr_samples[0][0], selected_slots_only=True))

    # DEBUG PRINT
    print('Partition:', len(trainset), len(validset), len(testset))
    print('Unique MRs in trainset:', len(train_mrs_slots_only))

    trainset_df = pd.DataFrame(trainset, columns=['mr', 'ref'])
    validset_df = pd.DataFrame(validset, columns=['mr', 'ref'])
    testset_df = pd.DataFrame(testset, columns=['mr', 'ref'])

    dataset_dir = os.path.dirname(dataset_filepath)
    trainset_df.to_csv(os.path.join(dataset_dir, 'train.csv'), index=False, encoding='utf8')
    validset_df.to_csv(os.path.join(dataset_dir, 'valid.csv'), index=False, encoding='utf8')
    testset_df.to_csv(os.path.join(dataset_dir, 'test.csv'), index=False, encoding='utf8')

    return trainset, validset, testset


def remove_vals_from_slots(mr, selected_slots_only=False):
    if selected_slots_only:
        # slots_to_remove_vals = ['name', 'release_year', 'exp_release_date', 'developer',
        #                         'player_perspective', 'platforms', 'esrb', 'rating',
        #                         'has_multiplayer', 'available_on_steam', 'has_linux_release', 'has_mac_release']
        slots_to_remove_vals = ['name', 'release_year', 'exp_release_date', 'developer',
                                'player_perspective', 'genres', 'platforms', 'esrb', 'has_multiplayer']

        for slot in slots_to_remove_vals:
            mr = re.sub(slot + r'\[.+?\]', slot, mr)
    else:
        mr = re.sub(r'\[.+?\]', '', mr)

    return mr


def print_stats_from_dict(stats_dict):
    for key, val in stats_dict.items():
        print(key + ': ' + str(val))


def main():
    mr_gen = MRGenerator('video_game')
    da = 'recommend'
    num_slots = 3

    # ----

    games_file_in = os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation', 'video_games.csv')
    mr_gen.create_mrs_from_game_data(games_file_in, da, num_slots)

    games_file_out_hit = os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation', 'video_games_mrs{0}_hit.csv'.format(
        ('_' + da) if da is not None else ''))
    shuffle_samples_csv(games_file_out_hit)

    # ----

    # file_in = os.path.join(config.VIDEO_GAME_DATA_DIR, 'results_combined_and_fixed.csv')
    # file_out = os.path.join(config.VIDEO_GAME_DATA_DIR, 'results_combined_and_fixed_mrs.txt')
    #
    # mr_gen.create_mrs_from_csv(file_in, file_out)

    # ----

    # file_dataset = os.path.join(config.VIDEO_GAME_DATA_DIR, 'dataset.csv')
    # file_misses = os.path.join(config.VIDEO_GAME_DATA_DIR, 'dataset_misses.csv')
    #
    # train_test_split(file_dataset, file_misses)

    # ----

    # dataset_path = os.path.join(config.VIDEO_GAME_DATA_DIR, 'train.csv')
    # shuffle_samples_csv(dataset_path)


if __name__ == '__main__':
    main()
