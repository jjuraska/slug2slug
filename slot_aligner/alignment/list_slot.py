import re
from nltk.tokenize import word_tokenize

from slot_aligner.alignment.utils import find_first_word_in_tok_text, get_slot_value_alternatives
from slot_aligner.alignment.categorical_slots import find_value_alternative


def align_list_slot(text, slot, value, match_all=True, mode='exact_match', item_sep='; '):
    """
    MR      := slot[value]
    value   := item || item; item;...
    item    := tok || tok tok...
    """
    leftmost_pos = -1

    # TODO: load alternatives only once
    alternatives = get_slot_value_alternatives(slot)

    # Preprocess the input text
    text = re.sub('-', ' ', text)
    text_tok = word_tokenize(text)

    # Split the slot value into individual items
    items = value.split(item_sep)

    # Search for all individual items exhaustively
    for item in items:
        pos = find_value_alternative(text, text_tok, item, alternatives, mode=mode)

        if match_all and pos < 0:
            return -1

        if leftmost_pos < 0 or 0 <= pos < leftmost_pos:
            leftmost_pos = pos

    print('leftmost:', leftmost_pos)

    return leftmost_pos


def align_list_with_conjunctions_slot(text, slot, value, match_all=True):
    separators = [',', 'and', 'with']

    value_tok = word_tokenize(value)
    value_items = []
    end_of_prev_item = -1
    leftmost_pos = -1

    # Split the value into items
    for i, tok in enumerate(value_tok):
        if tok in separators:
            item = ' '.join(value_tok[end_of_prev_item + 1:i])
            value_items.append(item)
            end_of_prev_item = i

    if end_of_prev_item < len(value_items) - 1:
        item = ' '.join(value_tok[end_of_prev_item + 1:])
        value_items.append(item)

    for item in value_items:
        pos = text.find(item)
        if leftmost_pos == -1 or pos < leftmost_pos:
            leftmost_pos = pos
        if match_all and pos < 0:
            return -1

    if leftmost_pos < 0:
        return -1

    return leftmost_pos
