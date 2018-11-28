# -*- coding: utf-8 -*-

import os
import io
import string
import re
import itertools
from collections import OrderedDict
from nltk.tokenize import word_tokenize, sent_tokenize

import config
from slot_aligner.alignment.boolean_slot import align_boolean_slot
from slot_aligner.alignment.list_slot import align_list_slot, align_list_with_conjunctions_slot
from slot_aligner.alignment.numeric_slot import align_numeric_slot_with_unit
from slot_aligner.alignment.scalar_slot import align_scalar_slot
from slot_aligner.alignment.categorical_slots import align_categorical_slot, foodSlot


customerrating_mapping = {
    'slot': 'rating',
    'values': {
        'low': 'poor',
        'average': 'average',
        'high': 'excellent',
        '1 out of 5': 'poor',
        '3 out of 5': 'average',
        '5 out of 5': 'excellent'
    }
}


def dontcareRealization(sent, slot, value):
    curr = re.sub('-', ' ', sent)
    curr = re.sub('\'', '', curr)
    curr = curr.lower()
    curr_tokens = word_tokenize(curr)

    slot_root = reduce_slot_name(slot)
    slot_root_plural = get_plural(slot_root)

    if slot_root in curr_tokens or slot_root_plural in curr_tokens or slot in curr_tokens:
        for x in ['any', 'all', 'vary', 'varying', 'varied', 'various', 'variety', 'different',
                  'unspecified', 'irrelevant', 'unnecessary', 'unknown', 'n/a', 'particular', 'specific', 'priority', 'choosy', 'picky',
                  'regardless', 'disregarding', 'disregard', 'excluding', 'unconcerned', 'matter', 'specification',
                  'concern', 'consideration', 'considerations', 'factoring', 'accounting', 'ignoring']:
            if x in curr_tokens:
                return True
        for x in ['no preference', 'no predetermined', 'no certain', 'wide range', 'may or may not',
                  'not an issue', 'not a factor', 'not important', 'not considered', 'not considering', 'not concerned',
                  'without a preference', 'without preference', 'without specification', 'without caring', 'without considering',
                  'not have a preference', 'dont have a preference', 'not consider', 'dont consider', 'not mind', 'dont mind',
                  'not caring', 'not care', 'dont care', 'didnt care']:
            if x in curr:
                return True
        if ('preference' in curr_tokens or 'specifics' in curr_tokens) and ('no' in curr_tokens):
            return True
    
    return False


def noneRealization(sent, slot, value):
    curr = re.sub('-', ' ', sent)
    curr = re.sub('\'', '', curr)
    curr = curr.lower()
    curr_tokens = word_tokenize(curr)
        
    if reduce_slot_name(slot) in curr_tokens:
        for x in ['information', 'info', 'inform', 'results', 'requirement', 'requirements', 'specification', 'specifications']:
            if x in curr_tokens and ('no' in curr_tokens or 'not' in curr_tokens or 'any' in curr_tokens):
                return True
    
    return False


def checkDelexSlots(slot, matches):
    for match in matches:
        if slot in match:
            return match

    return None


def reduce_slot_name(slot):
    slot = slot.replace('range', '')
    slot = slot.replace('rating', '')
    slot = slot.replace('size', '')

    if slot == 'hasusbport':
        slot = 'usb'
    elif slot == 'hdmiport':
        slot = 'hdmi'
    elif slot == 'powerconsumption':
        slot = 'power'
    elif slot == 'isforbusinesscomputing':
        slot = 'business'

    return slot.lower()


def get_plural(word):
    if word.endswith('y'):
        return re.sub(r'y$', 'ies', word)
    elif word.endswith('e'):
        return re.sub(r'e$', 'es', word)
    else:
        return word + 's'


def get_scalar_slots():
    return {
        'customerrating': {
            'low': 1,
            'average': 2,
            'high': 3,
            '1 out of 5': 1,
            '3 out of 5': 2,
            '5 out of 5': 3
        },
        'pricerange': {
            'high': 1,
            'moderate': 2,
            'cheap': 3,
            'more than £30': 1,
            '£20-25': 2,
            'less than £20': 3
        },
        'familyfriendly': {
            'no': 1,
            'yes': 3
        }
    }


# TODO: refactor the following 4 functions and encapsulate the repeating parts
def split_content(old_mrs, old_utterances, filename, use_heuristics=True, permute=False):
    """Splits the MRs into multiple MRs with the corresponding individual sentences.
    """

    # delex_slots = ['name', 'eattype', 'food', 'pricerange', 'customerrating', 'area', 'familyfriendly', 'near']
    new_mrs = []
    new_utterances = []
    
    slot_fails = OrderedDict()
    instance_fails = set()
    misses = ['The following samples were removed: ']
    
    base = max(int(len(old_utterances) * .1), 1)
    benchmarks = [base * i for i in range(1, 11)]

    for index in range(len(old_mrs)):
        # progress message
        if index in benchmarks:
            curr_state = index / base
            print('Slot alignment is ' + str(10 * curr_state) + '% done.')

        cur_mr = old_mrs[index]
        cur_utt = old_utterances[index]
        cur_utt = re.sub(r'\s+', ' ', cur_utt).strip()
        sents = sent_tokenize(cur_utt)
        root_sent = sents[0]

        new_pair = {sent: OrderedDict() for sent in sents}
        slots_found = set()
        rm_slot = []

        for slot, value in cur_mr.items():
            slot_root = slot.rstrip(string.digits)
            value = value.lower()
            has_slot = False

            # search for the realization of each token in each sentence
            for sent, new_slots in new_pair.items():
                found_slot = False
                sent = sent.lower()

                if slot_root == 'da':
                    found_slot = True
                elif value.lower() in sent:
                    found_slot = True
                elif slot_root == 'name':
                    sent_tokens = word_tokenize(sent)
                    for pronoun in ['it', 'its', 'it\'s', 'they', 'their', 'they\'re']:
                        if pronoun in sent_tokens:
                            found_slot = True
                            break
                elif use_heuristics:
                    # universal slot values
                    if value == 'dontcare':
                        if dontcareRealization(sent, slot_root, value):
                            found_slot = True
                    elif value == 'none':
                        if noneRealization(sent, slot_root, value):
                            found_slot = True

                    # E2E dataset slots
                    elif slot_root == 'eattype':
                        if align_categorical_slot(sent, slot_root, value, mode='first_word') >= 0:
                            found_slot = True
                    elif slot_root == 'food':
                        if foodSlot(sent, value) >= 0:
                            found_slot = True
                    elif slot_root == 'pricerange':
                        if align_scalar_slot(sent, slot_root, value, slot_stem_only=True) >= 0:
                            found_slot = True
                    elif slot_root == 'customerrating':
                        if align_scalar_slot(sent, slot_root, value,
                                             slot_mapping=customerrating_mapping['slot'],
                                             value_mapping=customerrating_mapping['values'],
                                             slot_stem_only=True) >= 0:
                            found_slot = True
                    elif slot_root == 'area':
                        if align_categorical_slot(sent, slot_root, value, mode='first_word') >= 0:
                            found_slot = True
                    elif slot_root == 'familyfriendly':
                        if align_boolean_slot(sent, slot_root, value) >= 0:
                            found_slot = True

                    # TV dataset slots
                    elif slot_root == 'type':
                        if align_categorical_slot(sent, slot_root, value, mode='first_word') >= 0:
                            found_slot = True
                    elif slot_root == 'hasusbport':
                        if align_boolean_slot(sent, slot_root, value, true_val='true', false_val='false') >= 0:
                            found_slot = True
                    elif slot_root in ['screensize', 'price', 'powerconsumption']:
                        if align_numeric_slot_with_unit(sent, slot_root, value) >= 0:
                            found_slot = True
                    elif slot_root in ['color', 'accessories']:
                        if align_list_with_conjunctions_slot(sent, slot_root, value) >= 0:
                            found_slot = True

                    # Laptop dataset slots
                    elif slot_root in ['weight', 'battery', 'drive', 'dimension']:
                        if align_numeric_slot_with_unit(sent, slot_root, value) >= 0:
                            found_slot = True
                    elif slot_root in ['design', 'utility']:
                        if align_list_with_conjunctions_slot(sent, slot_root, value, match_all=False) >= 0:
                            found_slot = True
                    elif slot_root == 'isforbusinesscomputing':
                        if align_boolean_slot(sent, slot_root, value, true_val='true', false_val='false') >= 0:
                            found_slot = True

                    # Video game dataset slots
                    elif slot_root == 'playerperspective':
                        if align_list_slot(sent, slot_root, value, match_all=False, mode='first_word') >= 0:
                            found_slot = True
                    elif slot_root == 'genres':
                        if align_list_slot(sent, slot_root, value, match_all=False, mode='exact_match') >= 0:
                            found_slot = True
                    elif slot_root == 'platforms':
                        if align_list_slot(sent, slot_root, value, match_all=False, mode='first_word') >= 0:
                            found_slot = True
                    elif slot_root in ['esrb', 'rating']:
                        if align_scalar_slot(sent, slot_root, value, slot_stem_only=False) >= 0:
                            found_slot = True
                    elif slot_root in ['hasmultiplayer', 'availableonsteam', 'haslinuxrelease', 'hasmacrelease']:
                        if align_boolean_slot(sent, slot_root, value) >= 0:
                            found_slot = True

                if found_slot:
                    new_slots[slot] = value
                    slots_found.add(slot)
                    has_slot = True

            if not has_slot:
                # if slot in ['eattype', 'familyfriendly', 'area', 'near', 'food']:
                misses.append('Couldn\'t find ' + slot + '(' + value + ') - ' + old_utterances[index])
                rm_slot.append(slot)
                # continue
                instance_fails.add(cur_utt)
                if slot not in slot_fails:
                    slot_fails[slot] = 0
                slot_fails[slot] += 1
        else:
            # remove slots whose realizations were not found
            for slot in rm_slot:
                del cur_mr[slot]
                
            new_mrs.append(cur_mr)
            new_utterances.append(cur_utt)
            
            if len(new_pair) > 1:
                for sent, new_slots in new_pair.items():
                    if root_sent == sent:
                        new_slots['position'] = 'outer'
                    else:
                        new_slots['position'] = 'inner'
                        
                    new_mrs.append(new_slots)
                    new_utterances.append(sent)
                    
            if permute:
                permuteSentCombos(new_pair, new_mrs, new_utterances, max_iter=True)

    misses.append('We had these misses from all categories: ' + str(slot_fails.items()))
    misses.append('So we had ' + str(len(instance_fails)) + ' samples with misses out of ' + str(len(old_utterances)))
    with io.open(os.path.join(config.SLOT_ALIGNER_DIR, '_logs', filename), 'w', encoding='utf8') as log_file:
        log_file.write('\n'.join(misses))

    return new_mrs, new_utterances


def score_alignment(curr_utterance, curr_mr, scoring="default+over-class"):
    """Scores a delexicalized utterance based on the rate of unrealized and/or overgenerated slots.
    """

    slots_found = set()
    sent = curr_utterance
    matches = set(re.findall(r'<slot_.*?>', sent))
    num_slot_overgens = 0

    for slot, value in curr_mr.items():
        slot_root = slot.rstrip(string.digits)
        value = value.lower()
        found_slot = False
        
        if slot_root == 'da':
            found_slot = True
        elif re.match(r'<!.*?>', slot_root):
            found_slot = True
        else:
            delex_slot = checkDelexSlots(slot, matches)
            if delex_slot:
                found_slot = True
                matches.remove(delex_slot)
            else:
                sent = sent.lower()
                if value in sent:
                    value_cnt = sent.count(value)
                    if value_cnt > 1:
                        num_slot_overgens += value_cnt - 1
                    found_slot = True
                elif value == 'dontcare':
                    if dontcareRealization(sent, slot_root, value):
                        slot_cnt = sent.count(reduce_slot_name(slot_root))
                        if slot_cnt > 1:
                            num_slot_overgens += slot_cnt - 1
                        found_slot = True
                elif value == 'none':
                    if noneRealization(sent, slot_root, value):
                        slot_cnt = sent.count(reduce_slot_name(slot_root))
                        if slot_cnt > 1:
                            num_slot_overgens += slot_cnt - 1
                        found_slot = True

                elif slot_root == 'name':
                    for pronoun in ['it', 'its', 'it\'s', 'they']:
                        if pronoun in word_tokenize(curr_utterance.lower()):
                            found_slot = True
                elif slot_root == 'pricerange':
                    if align_scalar_slot(sent, slot_root, value, slot_stem_only=True) >= 0:
                        found_slot = True
                elif slot_root == 'familyfriendly':
                    if align_boolean_slot(sent, slot_root, value) >= 0:
                        found_slot = True
                elif slot_root == 'food':
                    if foodSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == 'area':
                    if align_categorical_slot(sent, slot_root, value, mode='exact_match') >= 0:
                        found_slot = True
                elif slot_root == 'eattype':
                    if align_categorical_slot(sent, slot_root, value, mode='exact_match') >= 0:
                        found_slot = True
                elif slot_root == 'customerrating':
                    if align_scalar_slot(sent, slot_root, value,
                                         slot_mapping=customerrating_mapping['slot'],
                                         value_mapping=customerrating_mapping['values'],
                                         slot_stem_only=True) >= 0:
                        found_slot = True
                
                elif slot_root == 'type':
                    if align_categorical_slot(sent, slot_root, value, mode='exact_match') >= 0:
                        found_slot = True
                elif slot_root == 'hasusbport':
                    if align_boolean_slot(sent, slot_root, value, true_val='true', false_val='false') >= 0:
                        found_slot = True
                elif slot_root in ['screensize', 'price', 'powerconsumption']:
                    if align_numeric_slot_with_unit(sent, slot_root, value) >= 0:
                        found_slot = True
                elif slot_root in ['color', 'accessories']:
                    if align_list_with_conjunctions_slot(sent, slot_root, value) >= 0:
                        found_slot = True

                elif slot_root in ['weight', 'battery', 'drive', 'dimension']:
                    if align_numeric_slot_with_unit(sent, slot_root, value) >= 0:
                        found_slot = True
                elif slot_root in ['design', 'utility']:
                    if align_list_with_conjunctions_slot(sent, slot_root, value) >= 0:
                        found_slot = True
                elif slot_root == 'isforbusinesscomputing':
                    if align_boolean_slot(sent, slot_root, value, true_val='true', false_val='false') >= 0:
                        found_slot = True

                elif slot_root == 'playerperspective':
                    if align_list_slot(sent, slot_root, value, mode='first_word') >= 0:
                        found_slot = True
                elif slot_root == 'genres':
                    if align_list_slot(sent, slot_root, value, mode='exact_match') >= 0:
                        found_slot = True
                elif slot_root == 'platforms':
                    if align_list_slot(sent, slot_root, value, mode='first_word') >= 0:
                        found_slot = True
                elif slot_root in ['esrb', 'rating']:
                    if align_scalar_slot(sent, slot_root, value, slot_stem_only=False) >= 0:
                        found_slot = True
                elif slot_root in ['hasmultiplayer', 'availableonsteam', 'haslinuxrelease', 'hasmacrelease']:
                    if align_boolean_slot(sent, slot_root, value) >= 0:
                        found_slot = True

        if found_slot:
            slots_found.add(slot)

    # if scoring == "default":
    #    return len(slots_found) / len(curr_mr)
    # elif scoring == "default+over-class":
    #    return (len(slots_found) / len(curr_mr)) / (len(matches) + 1)

    # if scoring == "default":
    #    return len(slots_found) / len(curr_mr)
    # elif scoring == "default+over-class":
    #    return (len(slots_found) - len(matches) + 1) / (len(curr_mr) + 1)

    if scoring == "default":
        return 1 / (len(curr_mr) - len(slots_found) + num_slot_overgens + 1)
    elif scoring == "default+over-class":
        return 1 / (len(curr_mr) - len(slots_found) + num_slot_overgens + 1) / (len(matches) + 1)


def count_errors(utt, mr):
    """Counts unrealized and overgenerated slots in a lexicalized utterance.
    """

    non_categorical_slots = ['familyfriendly', 'pricerange', 'customerrating',
                             'rating', 'hasmultiplayer', 'availableonsteam', 'haslinuxrelease', 'hasmacrelease']

    slots_found = set()
    sent = utt
    matches = set(re.findall(r'<slot_.*?>', sent))
    num_slot_overgens = 0

    for slot, value in mr.items():
        slot_root = slot.rstrip(string.digits)
        found_slot = False

        if slot_root == 'da':
            found_slot = True
        elif re.match(r'<!.*?>', slot_root):
            found_slot = True
        else:
            delex_slot = checkDelexSlots(slot, matches)
            if delex_slot:
                found_slot = True
                matches.remove(delex_slot)
            else:
                sent = sent.lower()
                if value.lower() in sent:
                    value_cnt = sent.count(value.lower())
                    if slot_root not in non_categorical_slots and value_cnt > 1:
                        print('OVERGEN SLOT:', slot_root)
                        num_slot_overgens += value_cnt - 1
                    found_slot = True
                elif value == 'dontcare':
                    if dontcareRealization(sent, slot_root, value):
                        slot_cnt = sent.count(reduce_slot_name(slot_root))
                        if slot_cnt > 1:
                            num_slot_overgens += slot_cnt - 1
                        found_slot = True
                elif value == 'none':
                    if noneRealization(sent, slot_root, value):
                        slot_cnt = sent.count(reduce_slot_name(slot_root))
                        if slot_cnt > 1:
                            num_slot_overgens += slot_cnt - 1
                        found_slot = True

                elif slot_root == 'pricerange':
                    if align_scalar_slot(sent, slot_root, value, slot_stem_only=True) >= 0:
                        found_slot = True
                elif slot_root == 'familyfriendly':
                    if align_boolean_slot(sent, slot_root, value) >= 0:
                        found_slot = True
                elif slot_root == 'food':
                    if foodSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == 'area':
                    if align_categorical_slot(sent, slot_root, value, mode='exact_match') >= 0:
                        found_slot = True
                elif slot_root == 'eattype':
                    if align_categorical_slot(sent, slot_root, value, mode='exact_match') >= 0:
                        found_slot = True
                elif slot_root == 'customerrating':
                    if align_scalar_slot(sent, slot_root, value,
                                         slot_mapping=customerrating_mapping['slot'],
                                         value_mapping=customerrating_mapping['values'],
                                         slot_stem_only=True) >= 0:
                        found_slot = True

                elif slot_root == 'type':
                    if align_categorical_slot(sent, slot_root, value, mode='exact_match') >= 0:
                        found_slot = True
                elif slot_root == 'hasusbport':
                    if align_boolean_slot(sent, slot_root, value, true_val='true', false_val='false') >= 0:
                        found_slot = True
                elif slot_root in ['screensize', 'price', 'powerconsumption']:
                    if align_numeric_slot_with_unit(sent, slot_root, value) >= 0:
                        found_slot = True
                elif slot_root in ['color', 'accessories']:
                    if align_list_with_conjunctions_slot(sent, slot_root, value) >= 0:
                        found_slot = True

                elif slot_root in ['weight', 'battery', 'drive', 'dimension']:
                    if align_numeric_slot_with_unit(sent, slot_root, value) >= 0:
                        found_slot = True
                elif slot_root in ['design', 'utility']:
                    if align_list_with_conjunctions_slot(sent, slot_root, value) >= 0:
                        found_slot = True
                elif slot_root == 'isforbusinesscomputing':
                    if align_boolean_slot(sent, slot_root, value, true_val='true', false_val='false') >= 0:
                        found_slot = True

                elif slot_root == 'playerperspective':
                    if align_list_slot(sent, slot_root, value, mode='first_word') >= 0:
                        found_slot = True
                elif slot_root == 'genres':
                    if align_list_slot(sent, slot_root, value, mode='exact_match') >= 0:
                        found_slot = True
                elif slot_root == 'platforms':
                    if align_list_slot(sent, slot_root, value, mode='first_word') >= 0:
                        found_slot = True
                elif slot_root in ['esrb', 'rating']:
                    if align_scalar_slot(sent, slot_root, value, slot_stem_only=False) >= 0:
                        found_slot = True
                elif slot_root in ['hasmultiplayer', 'availableonsteam', 'haslinuxrelease', 'hasmacrelease']:
                    if align_boolean_slot(sent, slot_root, value) >= 0:
                        found_slot = True

        if found_slot:
            slots_found.add(slot)

    missing_slots = []
    for slot in mr:
        if slot not in slots_found:
            missing_slots.append(slot)

    num_erroneous_slots = (len(mr) - len(slots_found)) + num_slot_overgens

    return num_erroneous_slots, missing_slots


def find_alignment(utt, mr):
    """Identifies the mention position of each slot in the utterance.
    """

    alignment = []

    for slot, value in mr.items():
        slot_root = slot.rstrip(string.digits)
        utt = utt.lower()
        slot_pos = -1

        if slot_root == 'da':
            continue
        elif value.lower() in utt:
            slot_pos = utt.find(value.lower())
        # elif value == 'dontcare':
        #     if dontcareRealization(utt, slot_root, value):
        #         found_slot = True
        # elif value == 'none':
        #     if noneRealization(utt, slot_root, value):
        #         found_slot = True

        elif slot_root == 'eattype':
            slot_pos = align_categorical_slot(utt, slot_root, value, mode='exact_match')
        elif slot_root == 'food':
            slot_pos = foodSlot(utt, value)
        elif slot_root == 'pricerange':
            slot_pos = align_scalar_slot(utt, slot_root, value, slot_stem_only=True)
        elif slot_root == 'customerrating':
            slot_pos = align_scalar_slot(utt, slot_root, value,
                                         slot_mapping=customerrating_mapping['slot'],
                                         value_mapping=customerrating_mapping['values'],
                                         slot_stem_only=True)
        elif slot_root == 'area':
            slot_pos = align_categorical_slot(sent, slot_root, value, mode='exact_match')
        elif slot_root == 'familyfriendly':
            slot_pos = align_boolean_slot(utt, slot_root, value)

        elif slot_root == 'type':
            slot_pos = align_categorical_slot(utt, slot_root, value, mode='exact_match')
        elif slot_root == 'hasusbport':
            slot_pos = align_boolean_slot(utt, slot_root, value, true_val='true', false_val='false')
        elif slot_root in ['screensize', 'price', 'powerconsumption']:
            slot_pos = align_numeric_slot_with_unit(utt, slot_root, value)
        elif slot_root in ['color', 'accessories']:
            slot_pos = align_list_with_conjunctions_slot(utt, slot_root, value)

        elif slot_root in ['weight', 'battery', 'drive', 'dimension']:
            slot_pos = align_numeric_slot_with_unit(utt, slot_root, value)
        elif slot_root in ['design', 'utility']:
            slot_pos = align_list_with_conjunctions_slot(utt, slot_root, value)
        elif slot_root == 'isforbusinesscomputing':
            slot_pos = align_boolean_slot(utt, slot_root, value, true_val='true', false_val='false')

        elif slot_root == 'playerperspective':
            slot_pos = align_list_slot(utt, slot_root, value, mode='first_word')
        elif slot_root == 'genres':
            slot_pos = align_list_slot(utt, slot_root, value, mode='exact_match')
        elif slot_root == 'platforms':
            slot_pos = align_list_slot(utt, slot_root, value, mode='first_word')
        elif slot_root in ['esrb', 'rating']:
            slot_pos = align_scalar_slot(utt, slot_root, value, slot_stem_only=False)
        elif slot_root in ['hasmultiplayer', 'availableonsteam', 'haslinuxrelease', 'hasmacrelease']:
            slot_pos = align_boolean_slot(utt, slot_root, value)

        if slot_pos >= 0:
            alignment.append((slot_pos, slot, value))

    # Sort the slot realizations by their position
    alignment.sort(key=lambda x: x[0])

    return alignment


def mergeOrderedDicts(mrs, order=None):
    if order is None:
        order = ['da', 'name', 'eattype', 'food', 'pricerange', 'customerrating', 'area', 'familyfriendly', 'near',
                 'type', 'family', 'hasusbport', 'hdmiport', 'ecorating', 'screensizerange', 'screensize', 'pricerange', 'price', 'audio', 'resolution', 'powerconsumption', 'color', 'accessories', 'count',
                 'processor', 'memory', 'driverange', 'drive', 'batteryrating', 'battery', 'weightrange', 'weight', 'dimension', 'design', 'utility', 'platform', 'isforbusinesscomputing', 'warranty']
    merged_mr = OrderedDict()
    for slot in order:
        for mr in mrs:
            if slot in mr:
                merged_mr[slot] = mr[slot]
                break
    return merged_mr


def mergeEntries(merge_tuples):
    """
    :param merge_tuples: list of (utterance, mr) tuples to merge into one pair
    :return:
    """
    sent = ""
    mr = OrderedDict()
    mrs = []
    for curr_sent, curr_mr in merge_tuples:
        sent += " " + curr_sent
        mrs.append(curr_mr)
    mr = mergeOrderedDicts(mrs)
    return mr, sent


def permuteSentCombos(newPairs, mrs, utterances, max_iter=False, depth=1, assume_root=False):
    """
    :param newPairs: dict of {utterance:mr}
    :param mrs: mrs list - assume it's passed in
    :param utterances: utterance list - assume it's passed in
    :param depth: the depth of the combinations. 1 for example means a root sentence + one follow on.
        For example:
        utterance: a. b. c. d.
        depth 1, root a:
        a. b., a. c., a. d.
        depth 2, root a:
        a. b. c., a. d. c., ...
    :param assume_root: if we assume the first sentence in the list of sentences is the root most sentence, this is true,
        if this is true then we will only consider combinations with the the first sentence being the root.
        Note - a sentence is a "root" if it has the actual name of the restraunt in it. In many cases, there is only
        one root anyways.
    :return:
    """
    if len(newPairs) <= 1:
        return
    roots = []
    children = []
    for sent, new_slots in newPairs.items():
        if "name" in new_slots and new_slots["name"] in sent:
            roots.append((sent, new_slots))
        else:
            children.append((sent, new_slots))
    for root in roots:
        tmp = children + roots
        tmp.remove(root)

        combs = []
        for i in range(1, len(tmp) + 1):
            els = [list(x) for x in itertools.combinations(tmp, i)]
            combs.extend(els)

        if max_iter:
            depth = len(tmp)

        for comb in combs:
            if 0 < len(comb) <= depth:
                new_mr, new_utterance = mergeEntries([root] + comb)
                if "position" in new_mr:
                    del new_mr["position"]
                new_utterance = new_utterance.strip()
                if new_utterance not in utterances:
                    mrs.append(new_mr)
                    utterances.append(new_utterance)

        if assume_root:
            break
    # frivolous return for potential debug
    return utterances, mrs


# ---- UNIT TESTS ----

def testPermute():
    """Tests the permutation function.
    """

    newPairs = {"There is a pizza place named Chucky Cheese.": {"name": "Chucky Cheese"},
                "Chucky Cheese Sucks.": {"name": "Chucky Cheese"},
                "It has a ball pit.": {"b": 1}, "The mascot is a giant mouse.": {"a": 1}}

    utterances, mrs = permuteSentCombos(newPairs, [], [])

    for mr, utt in zip(mrs, utterances):
        print(mr, "---", utt)


# ---- MAIN ----

def main():
    print(foodSlot('There is a coffee shop serving good pasta.', 'Italian'))
    # print(foodSlot('This is a green tree.', 'Italian'))

    # testPermute()


if __name__ == '__main__':
    main()
