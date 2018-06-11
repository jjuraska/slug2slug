import os
import pandas as pd
import numpy as np
from collections import OrderedDict

import config


NUM_VARS = 10


def create_mrs_from_game_data(file_in, file_out):
    mrs = []

    # Read in the CSV file with the video game data
    df = pd.read_csv(file_in, header=0, dtype=object, encoding='utf8')

    # Initialize the MR length distribution dictionary
    mr_len_distr = OrderedDict()
    for i in range(3, 9):
        mr_len_distr[str(i)] = 0

    for _, row in df.iterrows():
        mr_var_cnt = 0
        while mr_var_cnt < NUM_VARS:
            # Generate a random MR from the current game's data
            mr_dict = get_random_slot_comb(row)

            # Assemble the MR in a string form
            mr = ''
            for slot, val in mr_dict.items():
                mr += slot + '[' + str(val) + '], '
            mr = mr[:-2]

            # If such an MR variant has not yet generated, save it
            if mr not in mrs[-NUM_VARS:]:
                # DEBUG PRINT
                # if len(mr_dict) > 8:
                #     print('CHECK:', mr_dict)

                mrs.append(mr)
                mr_len_distr[str(len(mr_dict))] = mr_len_distr.get(str(len(mr_dict)), 0) + 1
                mr_var_cnt += 1

            # The not-yet-released games typically have fewer slots, so fewer MR variants are possible for them
            if 'exp_release_date' in mr_dict and mr_var_cnt >= 4:
                break

    # Print MR length distribution
    print(mr_len_distr)

    # Store the generated MRs to file
    with open(file_out, 'w', encoding='utf8') as f_out:
        f_out.write('\n'.join(mrs))


def get_random_slot_comb(row):
    # Remove all NaN values from the row
    mr_dict = OrderedDict(row[~row.isnull()])

    # if 'exp_release_date' in mr_dict:
    #     return mr_dict

    max_slots = int(min(max(np.round(np.random.normal(loc=5.5, scale=1.5)), 3), 8))
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

        # Mark the current slot for removal
        mr_dict[slot] = None
        num_slots -= 1

        # Apply slot dependency rules
        if slot == 'platforms':
            for s in ['available_on_steam', 'has_linux_release', 'has_mac_release']:
                if s in mr_dict:
                    mr_dict[s] = None
                    num_slots -= 1
        elif slot == 'has_linux_release':
            if 'has_mac_release' in mr_dict:
                mr_dict['has_mac_release'] = None
                num_slots -= 1
        elif slot == 'has_mac_release':
            if 'has_linux_release' in mr_dict:
                mr_dict['has_linux_release'] = None
                num_slots -= 1

    # Remove the marked slots from the MR
    mr_dict_reduced = OrderedDict()
    for slot, val in mr_dict.items():
        if val:
            mr_dict_reduced[slot] = val

    return mr_dict_reduced


def main():
    games_file_in = os.path.join(config.DATA_DIR, 'video_game', 'video_games.csv')
    games_file_out = os.path.join(config.DATA_DIR, 'video_game', 'video_games_mrs.txt')

    create_mrs_from_game_data(games_file_in, games_file_out)


if __name__ == '__main__':
    main()
