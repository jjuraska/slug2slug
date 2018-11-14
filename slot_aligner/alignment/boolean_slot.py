import re
from nltk.tokenize import word_tokenize

from slot_aligner.alignment.utils import find_first_word_in_tok_text, find_all_in_list


NEG_IDX_PRE_THRESH = 10
NEG_POS_PRE_THRESH = 30
NEG_IDX_POST_THRESH = 10
NEG_POS_POST_THRESH = 30

negation_cues_pre = [
    'no', 'not', 'non', 'none', 'nor', 'never',
    'nt', 'isnt', 'arent', 'cant', 'cannot', 'doesnt', 'dont', 'didnt', 'wasnt', 'werent', 'wont',
    'excluded', 'lack', 'lacks', 'lacking', 'unavailable', 'without', 'zero',
    'everything but'
]
negation_cues_post = [
    'not', 'nor', 'never',
    'nt', 'isnt', 'arent', 'cant', 'cannot', 'doesnt', 'dont', 'didnt', 'wasnt', 'werent', 'wont',
    'excluded', 'unavailable',
    'out of luck'
]


def align_boolean_slot(text, slot, value, true_val='yes', false_val='no'):
    pos = -1

    text = re.sub('-', ' ', text)
    text = re.sub('\'', '', text)
    text_tok = word_tokenize(text)

    # Get the words that possibly realize the slot
    slot_stems = __get_boolean_slot_stems(slot)

    # Search for all possible slot realizations
    for slot_stem in slot_stems:
        idx, pos = find_first_word_in_tok_text(slot_stem, text_tok)
        if pos >= 0:
            if value == true_val:
                # Match an instance of the slot stem without a preceding negation
                if not __find_negation(text, text_tok, idx, pos, after=False):
                    return pos
            else:
                # Match an instance of the slot stem with a preceding or a following negation
                if __find_negation(text, text_tok, idx, pos, after=True):
                    return pos

    # If no match found and the value ~ False, search for alternative expressions of the opposite
    if pos < 0 and value == false_val:
        slot_antonyms = __get_boolean_slot_antonyms(slot)
        for slot_antonym in slot_antonyms:
            if ' ' in slot_antonym:
                pos = text.find(slot_antonym)
            else:
                _, pos = find_first_word_in_tok_text(slot_antonym, text_tok)

            if pos >= 0:
                return pos

    return -1


def __find_negation(text, text_tok, idx, pos, after=False):
    for negation in negation_cues_pre:
        if ' ' in negation:
            neg_pos = text.find(negation)
            if neg_pos >= 0:
                if 0 < (pos - neg_pos) < NEG_POS_PRE_THRESH:
                    return True
        else:
            neg_idxs = find_all_in_list(negation, text_tok)
            for neg_idx in neg_idxs:
                if 0 < (idx - neg_idx) < NEG_IDX_PRE_THRESH:
                    return True

    if after:
        for negation in negation_cues_post:
            if ' ' in negation:
                neg_pos = text.find(negation)
                if neg_pos >= 0:
                    if 0 < (neg_pos - pos) < NEG_POS_POST_THRESH:
                        return True
            else:
                neg_idxs = find_all_in_list(negation, text_tok)
                for neg_idx in neg_idxs:
                    if 0 < (neg_idx - idx) < NEG_IDX_POST_THRESH:
                        return True

    return False


def __get_boolean_slot_stems(slot):
    slot_stems = {
        'familyfriendly': ['family', 'families', 'kid', 'kids', 'child', 'children'],
        'hasusbport': ['usb'],
        'isforbusinesscomputing': ['business'],
        'hasmultiplayer': ['multiplayer', 'friends'],
        'availableonsteam': ['steam'],
        'haslinuxrelease': ['linux'],
        'hasmacrelease': ['mac']
    }

    return slot_stems.get(slot, [])


def __get_boolean_slot_antonyms(slot):
    slot_antonyms = {
        'familyfriendly': ['adult', 'adults'],
        'isforbusinesscomputing': ['personal', 'general', 'home', 'nonbusiness'],
        'hasmultiplayer': ['single player']
    }

    return slot_antonyms.get(slot, [])
