import os
import pandas as pd
import numpy as np
from collections import OrderedDict

import config


NUM_VARIATIONS = 10
slot_names = ['name',
              'release_year',
              'exp_release_date',
              'developer',
              'esrb',
              'rating',
              'genres',
              'player_perspective',
              'has_multiplayer',
              'platforms',
              'available_on_steam',
              'has_linux_release',
              'has_mac_release']


def create_mrs_from_game_data(file_in, file_out, file_out_hit=None):
    mrs = []
    mr_dicts = []

    # Read in the CSV file with the video game data
    df = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8')

    # Initialize the MR length distribution dictionary
    mr_len_distr = OrderedDict()
    for i in range(3, 9):
        mr_len_distr[str(i)] = 0

    # Initialize the slot distribution dictionary
    slot_distr = OrderedDict()
    for s in slot_names:
        slot_distr[s] = 0

    for _, row in df.iterrows():
        mr_var_cnt = 0
        while mr_var_cnt < NUM_VARIATIONS:
            # Generate a random MR from the current game's data
            mr_dict = get_random_slot_comb(row)

            # Assemble the MR in a string form
            mr = ''
            for slot, val in mr_dict.items():
                mr += slot + '[' + str(val) + '], '
            mr = mr[:-2]

            # If such an MR variant has not yet generated, save it
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
    with open(file_out, 'w', encoding='utf8') as f_out:
        f_out.write('\n'.join(mrs))

    # Format the generated MRs for a HIT and store them in a CSV file
    if file_out_hit is not None:
        save_csv_for_hit(mr_dicts, file_out_hit)


def create_mrs_from_csv(file_in, file_out):
    """Generates MRs in a textual form from an input CSV file in which each column corresponds to one slot.
    """

    mrs = []

    # Read in the CSV file with slot values in separate columns
    df = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8')

    # Initialize the MR length distribution dictionary
    mr_len_distr = OrderedDict()
    for i in range(3, 9):
        mr_len_distr[str(i)] = 0

    # Initialize the slot distribution dictionary
    slot_distr = OrderedDict()
    for s in slot_names:
        slot_distr[s] = 0

    for _, row in df.iterrows():
        # Remove all NaN values from the row
        mr_dict = OrderedDict(row[~row.isnull()])

        # Assemble the MR in a string form
        mr = ''
        for slot, val in mr_dict.items():
            mr += slot + '[' + str(val).replace(',', ';') + '], '
        mr = mr[:-2]

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


def get_random_slot_comb(row):
    pc_op_systems = ['has_linux_release', 'has_mac_release']
    pc_platforms = ['available_on_steam'] + pc_op_systems
    linux_mac_remove_flag = False

    # Remove all NaN values from the row
    mr_dict = OrderedDict(row[~row.isnull()])

    if 'developer' in mr_dict:
        if np.random.random() < 0.5:
            del mr_dict['developer']
    if 'esrb' in mr_dict:
        if np.random.random() < 0.5:
            del mr_dict['esrb']
    if 'has_multiplayer' in mr_dict:
        if np.random.random() < 0.25:
            del mr_dict['has_multiplayer']

    max_slots = int(min(max(np.round(np.random.normal(loc=6.5, scale=1.5)), 3), 8))
    max_slots = min(max_slots, len(mr_dict))
    num_slots = len(mr_dict)

    # Generate a random permutation of the slot indexes
    perm_idxs = np.random.permutation(num_slots)

    # DEBUG PRINT
    # print(num_slots)
    # print(perm_idxs)

    for slot_idx in perm_idxs:
        # Stop removing slots as soon as the MR has been reduced to the desired size
        if num_slots <= max_slots:
            break

        slot, val = list(mr_dict.items())[slot_idx]

        # Skip slots that were already removed due to one of the dependency rules (see below)
        if mr_dict[slot] is None:
            continue

        # Skip the mandatory slots
        if slot in ['name', 'genres', 'exp_release_date']:
            continue

        # Apply slot dependency rules
        if slot == 'platforms':
            pc_platform_slot_cnt = 0
            for s in pc_platforms:
                if mr_dict.get(s, None) is not None:
                    pc_platform_slot_cnt += 1

            if num_slots - pc_platform_slot_cnt - 1 >= 3:
                mr_dict['platforms'] = None
                num_slots -= 1
                for s in pc_platforms:
                    if mr_dict.get(s, None) is not None:
                        mr_dict[s] = None
                        num_slots -= 1
        elif slot in pc_op_systems:
            if num_slots - len(pc_op_systems) >= max_slots:
                if linux_mac_remove_flag:
                    for s in pc_op_systems:
                        if s in mr_dict:
                            mr_dict[s] = None
                            num_slots -= 1
                else:
                    linux_mac_remove_flag = True
        elif slot == 'available_on_steam':
            if np.random.random() < 0.5:
                continue
            else:
                mr_dict['available_on_steam'] = None
                num_slots -= 1
        else:
            # Mark the current slot for removal
            mr_dict[slot] = None
            num_slots -= 1

    # Remove the marked slots from the MR
    mr_dict_reduced = OrderedDict()
    for slot, val in mr_dict.items():
        if val:
            mr_dict_reduced[slot] = val

    return mr_dict_reduced


def save_csv_for_hit(mr_dicts, filepath):
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
    df_hit = pd.DataFrame(mr_dicts, columns=slot_names)
    df_hit.to_csv(filepath, index=False, encoding='utf8')


def shuffle_samples_csv(filepath):
    df = pd.read_csv(filepath, header=0, dtype=object, encoding='utf8')
    df = df.sample(frac=1).reset_index(drop=True)

    filepath_out = filepath.rstrip('.csv') + '_shuffled.csv'
    df.to_csv(filepath_out, index=False, encoding='utf8')


def print_stats_from_dict(stats_dict):
    for key, val in stats_dict.items():
        print(key + ': ' + str(val))


def main():
    # games_file_in = os.path.join(config.VIDEO_GAME_DATA_DIR, 'video_games.csv')
    # games_file_out = os.path.join(config.VIDEO_GAME_DATA_DIR, 'video_games_mrs.txt')
    # games_file_out_hit = os.path.join(config.VIDEO_GAME_DATA_DIR, 'video_games_mrs_hit.csv')

    # create_mrs_from_game_data(games_file_in, games_file_out, games_file_out_hit)
    # shuffle_samples_csv(games_file_out_hit)

    # ----

    file_in = os.path.join(config.VIDEO_GAME_DATA_DIR, 'results_combined_and_fixed.csv')
    file_out = os.path.join(config.VIDEO_GAME_DATA_DIR, 'results_combined_and_fixed_mrs.txt')

    create_mrs_from_csv(file_in, file_out)


if __name__ == '__main__':
    main()
