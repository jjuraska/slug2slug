import json

import config


def find_first_word_in_tok_text(w, t_tok):
    try:
        idx = t_tok.index(w)
    except ValueError as e:
        return -1, -1

    # Calculate approximate character position of the matched word
    pos = len(' '.join(t_tok[:idx]))

    return idx, pos


def find_all_in_list(val, lst):
    positions = []

    for pos, elem in enumerate(lst):
        if elem == val:
            positions.append(pos)

    return positions


def get_slot_value_alternatives(slot):
    with open(config.SLOT_ALIGNER_ALTERNATIVES, 'r') as f_alternatives:
        alternatives_dict = json.load(f_alternatives)

    return alternatives_dict.get(slot, {})
