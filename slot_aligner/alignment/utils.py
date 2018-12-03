import json

import config


def find_first_in_list(val, lst):
    idx = -1
    pos = -1

    for i, elem in enumerate(lst):
        if val == elem or val in elem.split('-') or val in elem.split('/'):
            idx = i

    if idx >= 0:
        # Calculate approximate character position of the matched value
        pos = len(' '.join(lst[:idx]))

    return idx, pos


def find_all_in_list(val, lst):
    indexes = []
    positions = []

    for i, elem in enumerate(lst):
        if val == elem or val in elem.split('-') or val in elem.split('/'):
            indexes.append(i)

            # Calculate approximate character position of the matched value
            positions.append(len(' '.join(lst[:i])))

    return indexes, positions


def get_slot_value_alternatives(slot):
    with open(config.SLOT_ALIGNER_ALTERNATIVES, 'r') as f_alternatives:
        alternatives_dict = json.load(f_alternatives)

    return alternatives_dict.get(slot, {})
