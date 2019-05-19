import os
import glob
import random
import re
import json
import pandas as pd
import numpy as np
from collections import OrderedDict

import config


class MRGenerator:
    def __init__(self, domain, num_variations, num_samples=None):
        self.domain = domain
        self.num_variations = num_variations
        self.num_samples = num_samples

        # Load slot info and constraints for the given domain
        if domain == 'video_game':
            with open(os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation', 'slots.json'), 'r') as f_slots:
                slots_dict = json.load(f_slots)
        else:
            raise ValueError('Unexpected domain "' + domain + '"')

        # All possible slots in the domain
        self.slots = slots_dict['names']
        # List slots in the domain
        self.list_slots = slots_dict['list_slots']
        # Mandatory and excluded slots for each DA
        self.da_constraints = slots_dict['da_constraints']
        # Value constraints for all slots in specific DAs, and for specific slots across all DAs (except for "inform")
        self.value_constraints = slots_dict['value_constraints']

    def create_mrs_from_game_data(self, file_in, da=None, num_slots=None):
        """Generates MRs by sampling the input data CSV file, creating two output files. The output text file contains
        a list of the MRs in their textual form. The output CSV file contains the MRs represented in a tabular form,
        including HTML formatting, and is ready to be used as an input file for an MTurk HIT.
        """

        mrs = []
        mr_dicts = []

        # Initialize the MR length distribution dictionary
        mr_len_distr = OrderedDict()
        for i in range(1, 9):
            mr_len_distr[str(i)] = 0

        # Initialize the slot distribution dictionary
        slot_distr = OrderedDict()
        for s in self.slots:
            slot_distr[s] = 0

        # Read in the CSV file with the video game data
        df = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8')

        for _, row in df.iterrows():
            # TEMP: skip future games
            if not pd.isnull(row['exp_release_date']):
                continue

            mr_var_cnt = 0
            while mr_var_cnt < self.num_variations:
                # Generate a random MR from the current game's data
                mr_dict = self.__get_random_slot_comb_for_da(row, da, num_slots)

                # Assemble the MR in a string form
                mr = mr_to_string(mr_dict, da)

                # If such an MR variant has not been generated yet, save it
                if mr not in mrs[-self.num_variations:]:
                    # DEBUG PRINT
                    # if len(mr_dict) > 8:
                    #     print('CHECK:', mr_dict)

                    mrs.append(mr)
                    mr_dicts.append(mr_dict)

                    mr_len_distr[str(len(mr_dict))] = mr_len_distr.get(str(len(mr_dict)), 0) + 1
                    # for slot in mr_dict:
                    #     slot_distr[slot] += 1

                    mr_var_cnt += 1

                # The not-yet-released games typically have fewer slots, so fewer MR variants are possible for them
                if 'exp_release_date' in mr_dict and mr_var_cnt >= 4:
                    break

        if self.num_samples is not None:
            # Take only a sample of a desired size from the generated MRs
            idxs = np.random.choice(np.arange(len(mrs)), self.num_samples, replace=False)
            print('MR sample indexes: ' + ', '.join(str(i) for i in idxs))
            mrs = np.array(mrs)[idxs].tolist()
            mr_dicts = np.array(mr_dicts)[idxs].tolist()
        else:
            # Print the MR length distribution and the slot distribution
            print('MR length distribution:\n')
            print_stats_from_dict(mr_len_distr)
            print('\nSlot distribution:\n')
            print_stats_from_dict(slot_distr)

        # Save the generated MRs to a text file
        file_out = os.path.splitext(file_in)[0] + '_mrs{0}.txt'.format(('_' + da) if da is not None else '')
        with open(file_out, 'w', encoding='utf8') as f_out:
            f_out.write('\n'.join(mrs))

        # Format the generated MRs for a HIT and save them in a CSV file
        file_out_hit = os.path.splitext(file_in)[0] + '_mrs{0} [HIT].csv'.format(('_' + da) if da is not None else '')
        self.__save_csv_for_hit(mr_dicts, file_out_hit)

    def create_mrs_from_csv(self, file_in, da=None, ignore_utt=True, create_hit_file=False):
        """Generates MRs in a textual form from an input CSV file in which each column corresponds to one slot.
        """

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

        # Read in the CSV file with slot values in separate columns
        df = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8')

        for _, row in df.iterrows():
            # Remove all NaN values from the row
            mr_dict = OrderedDict(row[~row.isnull()])

            if ignore_utt:
                mr_dict.pop('utterance', None)

            # Assemble the MR in a string form
            mr = mr_to_string(mr_dict, da)

            mrs.append(mr)
            mr_dicts.append(mr_dict)

            mr_len_distr[str(len(mr_dict))] = mr_len_distr.get(str(len(mr_dict)), 0) + 1
            for slot in mr_dict:
                slot_distr[slot] += 1

        # Print the MR length distribution and the slot distribution
        print('MR length distribution:\n')
        print_stats_from_dict(mr_len_distr)
        print('\nSlot distribution:\n')
        print_stats_from_dict(slot_distr)

        # Save the generated MRs to a text file
        file_out = os.path.splitext(file_in)[0].replace('processed_results', 'processed_mrs') + '.txt'
        with open(file_out, 'w', encoding='utf8') as f_out:
            f_out.write('\n'.join(mrs))

        if create_hit_file:
            # Format the generated MRs for a HIT and save them to a CSV file
            file_out_hit = os.path.splitext(file_in)[0].replace('processed_results', 'processed_mrs') + ' [HIT].csv'
            self.__save_csv_for_hit(mr_dicts, file_out_hit)

    def extract_hit_results_from_csv(self, file_in, use_specifier=False):
        """Extracts the columns with input content from an MTurk HIT results CSV file into the output file.
        The extracted slot-value pairs are aligned with the corresponding responses produced. The output CSV file
        has a similar format to the video game data file, only the slots are sparse and each MR has an utterance
        associated with it.
        """

        prefix_in = 'Input.'
        prefix_out = 'Answer.'
        suffix_utt = 'utterance'

        results = []

        # Read in the CSV file in the MTurk HIT format
        df = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8')

        for row_idx, row in df.iterrows():
            sample_data = {}
            for col_name in row.index:
                if col_name.startswith(prefix_in):
                    cell_content = row[col_name]
                    if not pd.isna(cell_content):
                        attr_name = re.match(prefix_in + r'(.+?)$', col_name).group(1)
                        sample_data[attr_name] = re.search(r'<b>(.+?)</b>', cell_content).group(1)

            sample_data['utterance'] = row[prefix_out + suffix_utt]
            results.append(sample_data)

        column_names = self.slots
        if use_specifier:
            column_names += ['specifier']
        column_names += ['utterance']

        if 'mturk_results' in file_in:
            file_out = file_in.replace('mturk_results', 'processed_results')
        else:
            file_out = os.path.splitext(file_in)[0] + ' NEW' + os.path.splitext(file_in)[1]

        df_results = pd.DataFrame(results, columns=column_names)
        df_results.to_csv(file_out, index=False, encoding='utf8')

    def extract_hit_results_from_csv_with_selection(self, file_in, num_responses=1, num_slots=1, use_specifier=False):
        """Extracts the columns with content (both given and produced) from an MTurk HIT results CSV file. By reading
        the values of the input fields, determines the slot-value pairs among the given data and extracts them into
        the output file. The extracted slot-value pairs are aligned with the corresponding responses produced. The
        output CSV file has a similar format to the video game data file, only the slots are sparse and each MR has
        an utterance associated with it.
        """

        prefix_in = 'Input.'
        prefix_out = 'Answer.'
        suffix_slot = 'selected_attr'
        suffix_utt = 'utterance'
        suffix_spec = 'specifier'

        attr_name_map = {
            'Release year': 'release_year',
            'Developer': 'developer',
            'ESRB content rating': 'esrb',
            'Genres': 'genres',
            'Player perspective': 'player_perspective',
            'Multiplayer': 'has_multiplayer',
            'Platform': 'platforms',
            'Availability on Steam': 'available_on_steam',
            'Linux': 'has_linux_release',
            'Mac': 'has_mac_release'
        }

        results = []

        # Read in the CSV file in the MTurk HIT format
        df = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8')

        for row_idx, row in df.iterrows():
            for i in range(1, num_responses + 1):
                sample_data = {}
                if num_slots > 0:
                    for j in range(1, num_slots + 1):
                        col_name = prefix_out + suffix_slot + str(i) + str(j)

                        try:
                            slot_selected = row[col_name]
                        except KeyError:
                            print('Warning: column "' + col_name + '" not found.')
                            continue

                        cell_content = row[prefix_in + str(slot_selected)]
                        if pd.isna(cell_content):
                            print('Warning: row ' + str(row_idx + 1) + ', response ' + str(i) + ', slot ' + str(j) + '\t>> ' + 'slot "' + slot_selected + '" not found.')
                            continue

                        val = re.search(r'<b>(.+?)</b>', cell_content)
                        sample_data[slot_selected] = val.group(1)
                else:
                    slot_name = str(row[prefix_in + 'attribute'])
                    slot_norm = attr_name_map[slot_name]
                    sample_data[slot_norm] = ' '

                if use_specifier:
                    sample_data['specifier'] = row[prefix_out + suffix_spec + str(i)]

                sample_data['utterance'] = row[prefix_out + suffix_utt + str(i)]
                results.append(sample_data)

                # DEBUG PRINT
                # print(sample_data)

        column_names = self.slots
        if use_specifier:
            column_names += ['specifier']
        column_names += ['utterance']

        if 'mturk_results' in file_in:
            file_out = file_in.replace('mturk_results', 'processed_results')
        else:
            file_out = os.path.splitext(file_in)[0] + ' NEW' + os.path.splitext(file_in)[1]

        df_results = pd.DataFrame(results, columns=column_names)
        df_results.to_csv(file_out, index=False, encoding='utf8')

    def merge_data_files(self, dataset, file1, file2):
        """"""

        # Read in the data
        df1 = pd.read_csv(os.path.join(config.DATA_DIR, dataset, file1), header=0, encoding='utf8')
        if file2 is not None:
            df2 = pd.read_csv(os.path.join(config.DATA_DIR, dataset, file2), header=0, encoding='utf8')

        if file2 is not None:
            df_merged = df1.append(df2, ignore_index=True).sort_values(by='mr')
        else:
            df_merged = df1.sort_values(by='mr')

        if 'final' in file1:
            file_out = re.sub(r', round \d', '', file1.replace('final', 'da'))
        else:
            file_out = os.path.splitext(file1)[0] + ' MERGED' + os.path.splitext(file1)[1]

        df_merged.to_csv(os.path.join(config.DATA_DIR, dataset, file_out), index=False, encoding='utf8')

    def merge_data_files_with_suffix(self, dataset, folder, suffix=''):
        """"""

        # Read all files with the given suffix in their name
        file_paths = glob.glob(os.path.join(config.DATA_DIR, dataset, folder, '*' + suffix + '.csv'))

        # Combine all files into a single DataFrame
        df_merged = pd.concat((pd.read_csv(f) for f in file_paths), ignore_index=True)

        # Shuffle rows
        df_merged = df_merged.sample(frac=1).reset_index(drop=True)

        # Group rows by MRs and expand them back
        df_grouped = pd.concat([utt_group for mr, utt_group in df_merged.groupby('mr', sort=False)], ignore_index=True)

        if suffix is not None and len(suffix) > 0:
            file_out = os.path.join(config.DATA_DIR, dataset, folder, 'merged_' + suffix + '.csv')
        else:
            file_out = os.path.join(config.DATA_DIR, dataset, folder, 'merged.csv')

        df_grouped.to_csv(file_out, index=False, encoding='utf8')

    def __get_random_slot_comb_for_da(self, row, da='inform', num_slots=None):
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

            if num_slots is not None:
                # Sample additional slots from the remaining slots
                num_additional_slots = num_slots - len(mandatory_slots)
                if num_additional_slots > 0:
                    mr_dict.update(random.sample(slot_value_dict.items(), num_additional_slots))
            else:
                # Add all the remaining non-excluded slots
                mr_dict.update(slot_value_dict)

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
            if val is not None:
                mr_dict[slot] = val

        return mr_dict

    def __load_slots_for_da(self, da):
        try:
            mandatory_slots = self.da_constraints[da]['mandatory']
        except KeyError:
            mandatory_slots = []

        try:
            excluded_slots = self.da_constraints[da]['excluded']
        except KeyError:
            excluded_slots = []

        return mandatory_slots, excluded_slots

    def __adjust_mr_for_da(self, da, mr_dict):
        """Applies predefined constraints to the slots' values. The constraints concern the number of elements
        in list values. The constraints that apply to all slots across a specific DA have a higher priority than
        the constraints that apply to a certain slot across all DAs.
        """

        if da in self.value_constraints.get('dialogue_acts', {}):
            # Get the value length constraint for the given DA
            max_num_values = self.value_constraints['dialogue_acts'][da]
            if max_num_values > 0:
                # Sample elements from the slot's value, whenever the value is a list
                for slot, val in mr_dict.items():
                    if slot in self.list_slots:
                        val = self.__value_to_list(val)
                        if max_num_values < len(val):
                            val = random.sample(val, max_num_values)
                        mr_dict[slot] = self.__value_to_str(val)
            else:
                # Remove all slots' values
                for slot in mr_dict:
                    mr_dict[slot] = None
        else:
            for slot, val in mr_dict.items():
                if slot in self.list_slots:
                    val = self.__value_to_list(val)
                    if slot in self.value_constraints.get('slots', {}):
                        # Get the value length constraint for the given slot
                        max_num_values = self.value_constraints['slots'][slot]
                        if max_num_values < 1:
                            # Remove the slot's value entirely
                            val = None
                        elif max_num_values < len(val):
                            # Sample elements from the slot's value, if the value is a list
                            val = random.sample(val, max_num_values)
                    mr_dict[slot] = self.__value_to_str(val)

        # New DA-specific slots
        if da == 'request':
            # Prepend an empty "specifier" slot to the beginning of the MR dict
            mr_dict['specifier'] = None
            mr_dict.move_to_end('specifier', last=False)

        # Apply domain-specific constraints to the slots in the MR
        self.__apply_domain_rules_to_mr(mr_dict)

    def __apply_domain_rules_to_mr(self, mr_dict):
        """Validates domain-specific constraints between slots in the given MR, and removes slots that do not
        satisfy the conditions.
        """

        if self.domain == 'video_game':
            pc_platforms = ['available_on_steam', 'has_linux_release', 'has_mac_release']
            # If PC not among platforms, drop the PC-related slots
            if mr_dict.get('platforms', None) is not None and 'pc' not in mr_dict['platforms'].lower():
                for p in pc_platforms:
                    mr_dict.pop(p, None)

    def __value_to_list(self, val_as_str):
        """Splits the value string around commas."""

        if val_as_str is None:
            return None

        return [elem.strip() for elem in val_as_str.split(',')]

    def __value_to_str(self, val_as_list):
        """Joins the list elements using a semicolon."""

        if val_as_list is None:
            return None

        return ', '.join(val_as_list)

    def __save_csv_for_hit(self, mr_dicts, filepath):
        # Add HTML formatting to the values
        for mr_dict in mr_dicts:
            for slot, val in mr_dict.items():
                if val is None:
                    val = ''

                if slot == 'name':
                    mr_dict[slot] = '<b>' + val + '</b>'
                elif slot == 'specifier':
                    mr_dict[slot] = 'Specifier: ' + '<b>' + val + '</b>'
                elif slot == 'release_year':
                    mr_dict[slot] = 'Year: ' + '<b>' + val + '</b>'
                elif slot == 'exp_release_date':
                    mr_dict[slot] = 'Expected release date: ' + '<b>' + val + '</b>'
                elif slot == 'developer':
                    mr_dict[slot] = 'Developer: ' + '<b>' + val + '</b>'
                elif slot == 'esrb':
                    mr_dict[slot] = 'ESRB content rating: ' + '<b>' + val + '</b>'
                elif slot == 'rating':
                    # mr_dict[slot] = 'Liking/Rating: ' + '<b>' + val + '</b>'
                    mr_dict[slot] = 'Liking: ' + '<b>' + val + '</b>'
                elif slot == 'genres':
                    mr_dict[slot] = 'Genres: ' + '<b>' + val + '</b>'
                elif slot == 'player_perspective':
                    mr_dict[slot] = 'Player perspective: ' + '<b>' + val + '</b>'
                elif slot == 'has_multiplayer':
                    mr_dict[slot] = 'Has multiplayer: ' + '<b>' + val + '</b>'
                elif slot == 'platforms':
                    # mr_dict[slot] = 'Platforms: ' + '<b>' + val + '</b>'
                    mr_dict[slot] = 'Platform: ' + '<b>' + val + '</b>'
                elif slot == 'available_on_steam':
                    mr_dict[slot] = 'Available on Steam: ' + '<b>' + val + '</b>'
                elif slot == 'has_linux_release':
                    mr_dict[slot] = 'Has a Linux release: ' + '<b>' + val + '</b>'
                elif slot == 'has_mac_release':
                    mr_dict[slot] = 'Has a Mac release: ' + '<b>' + val + '</b>'

        if 'specifier' in mr_dicts[0]:
            self.slots.append('specifier')

        # Store the formatted MRs to a CSV file
        df_hit = pd.DataFrame(mr_dicts, columns=self.slots)
        df_hit.to_csv(filepath, index=False, encoding='utf8')


# TODO: replace all uses of this function with the one in the data_loader.py file
def mr_to_string(mr_dict, da=None):
    slot_value_pairs = []

    for slot, val in mr_dict.items():
        slot_value_pairs.append(slot + '[{0}]'.format(str(val.strip()) if val is not None else ''))

    mr = ', '.join(slot_value_pairs)

    if da is not None:
        # Prepend the DA, and enclose the list of the MR's slot-value pairs in parentheses
        mr = da + '(' + mr + ')'

    return mr


def shuffle_samples_csv(filepath):
    df = pd.read_csv(filepath, header=0, dtype=object, encoding='utf8')
    df = df.sample(frac=1).reset_index(drop=True)

    filepath_out = filepath.rstrip('.csv') + '_shuffled.csv'
    df.to_csv(filepath_out, index=False, encoding='utf8')


def train_test_split(dataset_filepath, split_ratios, num_refs_per_mr=1, misses_filepath=None):
    assert isinstance(split_ratios, list) and sum(split_ratios) == 1

    valid_ratio = split_ratios[1]
    test_ratio = split_ratios[2]

    non_train_ratio = valid_ratio + test_ratio

    trainset = []
    validset = []
    testset = []

    train_mrs_slots_only = set()
    non_train_mrs_slots_only = set()
    # train_names = set()
    # non_train_names = set()

    df_dataset = pd.read_csv(dataset_filepath, header=0, encoding='utf8')
    num_samples = df_dataset.shape[0]

    if misses_filepath is not None:
        df_misses = pd.read_csv(misses_filepath, header=0, encoding='utf8')
        if df_misses.shape[0] != num_samples:
            raise ValueError('Input files do not have matching numbers of lines.')

    # DEBUG PRINT
    # print(num_samples)
    # print(df_misses.dtypes)

    cur_row = 0
    for mr_idx in range(num_samples // num_refs_per_mr):
        mr_samples = []
        has_ref_with_miss = False
        for ref_idx in range(num_refs_per_mr):
            mr_samples.append(tuple(df_dataset.iloc[cur_row, :].values))
            if misses_filepath is not None and df_misses.iat[cur_row, 2] > 0:
                has_ref_with_miss = True
            cur_row += 1

        # DEBUG PRINT
        # print(cur_row, '->', has_ref_with_miss)
        # print(mr_samples)

        # Remove the values of certain slots for determining MR similarity
        cur_mr_generalized = remove_vals_from_slots(mr_samples[0][0], selected_slots_only=True)

        # Extract the value of the name slot
        # cur_name = extract_name_from_mr_string(mr_samples[0][0])

        # Partition the samples into train/valid/test
        if cur_mr_generalized in non_train_mrs_slots_only or \
                cur_mr_generalized not in train_mrs_slots_only and \
                not has_ref_with_miss and random.random() < non_train_ratio:
            if random.random() < (valid_ratio / non_train_ratio):
                validset.extend(mr_samples)
            else:
                testset.extend(mr_samples)

            non_train_mrs_slots_only.add(cur_mr_generalized)
            # if cur_name is not None:
            #     non_train_names.add(cur_name)
        else:
            # All samples with missing/hallucinated information go to trainset
            trainset.extend(mr_samples)

            train_mrs_slots_only.add(cur_mr_generalized)
            # if cur_name is not None:
            #     train_names.add(cur_name)

    # DEBUG PRINT
    print('Partition:', len(trainset), len(validset), len(testset))
    print('Unique MRs in trainset:', len(train_mrs_slots_only))
    # print('Names in trainset:', train_names)

    trainset_df = pd.DataFrame(trainset, columns=['mr', 'ref'])
    validset_df = pd.DataFrame(validset, columns=['mr', 'ref'])
    testset_df = pd.DataFrame(testset, columns=['mr', 'ref'])

    dataset_dir = os.path.dirname(dataset_filepath)
    trainset_df.to_csv(os.path.join(dataset_dir, 'train.csv'), index=False, encoding='utf8')
    validset_df.to_csv(os.path.join(dataset_dir, 'valid.csv'), index=False, encoding='utf8')
    testset_df.to_csv(os.path.join(dataset_dir, 'test.csv'), index=False, encoding='utf8')

    return trainset, validset, testset


def extract_name_from_mr_string(mr, slot_name='name'):
    try:
        return re.search(slot_name + r'\[(.+?)\]', mr).group(1)
    except (AttributeError, IndexError):
        return None


# TODO: make more universal so it would work with different value separators
def remove_vals_from_slots(mr, selected_slots_only=False):
    if selected_slots_only:
        # slots_to_remove_vals = ['name', 'release_year', 'exp_release_date', 'developer',
        #                         'player_perspective', 'platforms', 'esrb', 'rating',
        #                         'has_multiplayer', 'available_on_steam', 'has_linux_release', 'has_mac_release']
        slots_to_remove_vals = ['name', 'developer']

        for slot in slots_to_remove_vals:
            mr = re.sub(slot + r'\[.+?\]', slot, mr)
    else:
        mr = re.sub(r'\[.+?\]', '', mr)

    return mr


def print_stats_from_dict(stats_dict):
    for key, val in stats_dict.items():
        print(key + ': ' + str(val))


def main():
    # Parameters for "inform" DA
    # domain = 'video_game'
    # num_variations = 10
    # num_total_samples = None
    # da = 'inform'
    # num_slots = None

    # Parameters for other DAs
    domain = 'video_game'
    num_variations = 1
    num_total_samples = 20
    da = 'request'
    num_slots = None

    mr_gen = MRGenerator(domain, num_variations, num_total_samples)

    # ----

    # games_file_in = os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation', 'video_games.csv')
    # mr_gen.create_mrs_from_game_data(games_file_in, da, num_slots)

    # ----

    # games_file_out_hit = os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation', 'video_games_mrs{0}_hit.csv'.format(
    #     ('_' + da) if da is not None else ''))
    # shuffle_samples_csv(games_file_out_hit)

    # ----

    # mturk_results_file_in = os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation',
    #                                      'video_games_mturk_results_round2_request_attribute (1 slot).csv')
    # mr_gen.extract_hit_results_from_csv_with_selection(mturk_results_file_in, num_responses=5, num_slots=0, use_specifier=False)

    # ----

    mturk_results_file_in = os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation',
                                         'video_games_mturk_results_inform.csv')
    mr_gen.extract_hit_results_from_csv(mturk_results_file_in, use_specifier=False)

    # ----

    # file_in = os.path.join(config.VIDEO_GAME_DATA_DIR, 'generation',
    #                        'video_games_processed_results_verify_attribute (4 slots).csv')
    # mr_gen.create_mrs_from_csv(file_in, da='verify_attribute', ignore_utt=True, create_hit_file=True)

    # ----

    # file1 = 'generation/video_games_final_verify_attribute (4 slots, round 1).csv'
    # file2 = 'generation/video_games_final_verify_attribute (4 slots, round 2).csv'
    # mr_gen.merge_data_files('video_game', file1, file2)

    # ----

    # file_dataset = os.path.join(config.VIDEO_GAME_DATA_DIR, 'individual_das', 'video_games_da_inform.csv')
    # file_misses = os.path.join(config.VIDEO_GAME_DATA_DIR, 'individual_das', 'video_games_da_inform (errors).csv')
    #
    # train_test_split(file_dataset, split_ratios=[0.8, 0.1, 0.1], num_refs_per_mr=3, misses_filepath=file_misses)

    # ----

    # mr_gen.merge_data_files_with_suffix('video_game', 'train-valid-test', suffix='train')

    # ----

    # dataset_path = os.path.join(config.VIDEO_GAME_DATA_DIR, 'train.csv')
    # shuffle_samples_csv(dataset_path)


if __name__ == '__main__':
    main()
