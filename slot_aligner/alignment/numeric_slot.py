import re
from nltk.tokenize import word_tokenize

from slot_aligner.alignment.utils import find_first_word_in_tok_text


def align_numeric_slot_with_unit(text, slot, value):
    text = re.sub('-', ' ', text)
    text = re.sub('\'', '', text)
    text_tok = word_tokenize(text)

    value_number = value.split(' ')[0]
    try:
        float(value_number)
    except ValueError:
        return -1

    _, pos = find_first_word_in_tok_text(value_number, text_tok)
    if pos >= 0:
        return pos

    return -1
