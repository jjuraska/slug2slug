import re
from nltk.tokenize import word_tokenize

from slot_aligner.alignment.utils import find_first_word_in_tok_text, get_slot_value_alternatives


def align_list_slot(text, slot, value, match_all=True, mode='exact_match', item_sep='; '):
    """
    MR      := slot[value]
    value   := item || item; item;...
    item    := tok || tok tok...
    """
    leftmost_pos = -1

    alternatives = get_slot_value_alternatives(slot)

    # Preprocess the input text
    text = re.sub('-', ' ', text)
    text_tok = word_tokenize(text)

    # Split the slot value into individual items
    items = value.split(item_sep)

    # Search for all individual items exhaustively
    for item in items:
        item_matched = False

        # Parse the item into tokens according to the selected mode
        if mode == 'first_word':
            item_alternatives = [item.split(' ')[0]]    # Single-element list
        elif mode == 'any_word':
            item_alternatives = item.split(' ')         # List of elements
        elif mode == 'all_words':
            item_alternatives = [item.split(' ')]       # List of single-element lists
        else:
            item_alternatives = [item]                  # Single-element list

        # Merge the tokens with the item's alternatives
        if item in alternatives:
            item_alternatives += alternatives[item]

        # Iterate over individual tokens of the item
        for item_alt in item_alternatives:
            # If the item is composed of a single token, convert it to a single-element list
            if not isinstance(item_alt, list):
                item_alt = [item_alt]

            # Keep track of the positions of all the item's tokens
            positions = []
            for tok in item_alt:
                if len(tok) > 4 or ' ' in tok:
                    # Search for multi-word values in the string representation
                    pos = text.find(tok)
                else:
                    # Search for single-word values in the tokenized representation
                    _, pos = find_first_word_in_tok_text(tok, text_tok)
                positions.append(pos)

            if all([p >= 0 for p in positions]):
                item_matched = True
                if leftmost_pos == -1 or min(positions) < leftmost_pos:
                    leftmost_pos = min(positions)
                break

        if match_all and not item_matched:
            return -1

    if not match_all and leftmost_pos < 0:
        return -1

    return leftmost_pos
