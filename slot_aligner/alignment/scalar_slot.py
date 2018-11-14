import re
from nltk.tokenize import word_tokenize

from slot_aligner.alignment.utils import find_first_word_in_tok_text, get_slot_value_alternatives


DIST_IDX_THRESH = 10
DIST_POS_THRESH = 30


def align_scalar_slot(text, slot, value, slot_mapping=None, value_mapping=None, slot_stem_only=False):
    slot_stem_idx = -1
    slot_stem_pos = -1

    text = re.sub('-', ' ', text)
    text = re.sub('\'', '', text)
    text_tok = word_tokenize(text)

    # Get the words that possibly realize the slot
    slot_stems = __get_scalar_slot_stems(slot)

    if slot_mapping is not None:
        slot = slot_mapping
    alternatives = get_slot_value_alternatives(slot)

    # Get the value's alternative realizations
    value_alternatives = [value]
    if value_mapping is not None:
        value = value_mapping[value]
        value_alternatives.append(value)
    if value in alternatives:
        value_alternatives += alternatives[value]

    # Search for all possible slot realizations
    for slot_stem in slot_stems:
        if len(slot_stem) == 1 and not slot_stem.isalnum():
            # Exception for single-letter special-character slot stems
            slot_stem_pos = text.find(slot_stem)
        elif len(slot_stem) > 4 or ' ' in slot_stem:
            slot_stem_pos = text.find(slot_stem)
        else:
            slot_stem_idx, slot_stem_pos = find_first_word_in_tok_text(slot_stem, text_tok)

        if slot_stem_pos >= 0:
            break

    if slot_stem_only and slot_stem_pos >= 0:
        return slot_stem_pos

    # Search for all possible value equivalents
    for val in value_alternatives:
        if len(val) > 4 or ' ' in val:
            # Search for multi-word values in the string representation
            pos = text.find(val)
            if pos >= 0:
                if slot_stem_pos >= 0:
                    # If the slot stem was found, make sure it's not too far from the value realization
                    if abs(pos - slot_stem_pos) < DIST_POS_THRESH:
                        return pos
                else:
                    return pos
        else:
            # Search for single-word values in the tokenized representation
            idx, pos = find_first_word_in_tok_text(val, text_tok)
            if pos >= 0:
                if slot_stem_pos >= 0:
                    if slot_stem_idx >= 0:
                        if abs(idx - slot_stem_idx) < DIST_IDX_THRESH:
                            return pos
                    else:
                        if abs(pos - slot_stem_pos) < DIST_POS_THRESH:
                            return pos
                else:
                    return pos

    return -1


def __get_scalar_slot_stems(slot):
    slot_stems = {
        'esrb': ['esrb'],
        'rating': ['rating', 'rated', 'rate', 'review', 'reviews'],
        'customerrating': ['customer', 'rating', 'rated', 'rate', 'review', 'reviews', 'star', 'stars'],
        'pricerange': ['price', 'pricing', 'cost', 'costs', 'dollars', 'pounds', 'euros', '$', '£', '€']
    }

    return slot_stems.get(slot, [])
