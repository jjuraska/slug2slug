# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import sys
import os
import io
import string
import json
import copy
import pandas as pd
import numpy as np
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
import re
import itertools

import data_loader


# TODO 139 instances in training which have the slot, but no value in the utterance
# TODO verify this is true... can we just delete this arg?
def familyFriendlySlot(sent, value):
    pos = -1

    # family-friendly, family friendly, kid-friendly, kid friendly, child(ren)-friendly, child(ren) friendly
    curr = re.sub('-', ' ', sent)
    curr = re.sub('\'', '', curr)
    curr = curr.lower()
    curr_tokens = word_tokenize(curr.lower())

    for sval in ['famil', 'kid', 'child']:
        pos = curr.find(sval)
        if pos >= 0:
            # root_sval = sval.split(' ')
            # curr_index = -1
            # neg_indicies = -1
            # for x, tok in enumerate(curr_tokens):
            #     if sval in tok:
            # if root_sval[0] == tok:
            # break
            # elif tok in ['not', 'non']:
            #     neg_indicies = x
            # if (value == 'no' and neg_indicies != -1 and curr_index != -1) or value == 'yes':
            # sval = ' '.join(curr_tokens[neg_indicies:curr_index+2])
            if value == 'no':
                for x in ['not', 'non', 'isnt', 'dont', 'doesnt', 'lowly', 'lacking', 'bad', 'none']:
                    if x in curr_tokens:
                        return pos
                return -1
            else:
                return pos
        elif value == 'no' and 'adult' in sent:
            return curr.find('adult')
    
    # print(curr)
    return pos


# TODO 166 instances in training which have the slot, but no value in the utterance or worse, city center is tn utterance
# TODO verify this is true... can we just delete this arg? It seems maybe in rl city center is near the river, hence they are interchangable?
def areaSlot(sent, value):
    """
    :param sent: target utterance
    :param value: slot value
    :return:
        Note it's only possible to have city centre and riverside as possible values...
    """
    pos = -1

    if value == 'riverside':
        return sent.find('river')
    elif value == 'city centre':
        for val in ['center', 'centre']:
            pos = sent.find(val)
            if pos >= 0:
                return pos

        if 'middle' in sent:
            for val in ['city', 'town']:
                pos = sent.find(val)
                if pos >= 0:
                    return pos

    return pos


def scorePriceRangeNaive(sent, value):
    pos = -1
    pot_values = {'cheap': 'less than £20',
                  'moderate': '£20-25',
                  'high': 'more than £30'}

    for k, v in pot_values.items():
        if value == k or value == v:
            pos = sent.find(k)
            if pos >= 0:
                return pos

            pos = sent.find(v)
            if pos >= 0:
                return pos

    return pos


def scoreCustomerRatingNaive(sent, value):
    pos = -1
    pot_values = {'1 out of 5': 'low',
                  '3 out of 5': 'average',
                  '5 out of 5': 'high'}

    if 'customer' in sent or 'rate' in sent or 'rating' in sent:
        for k, v in pot_values.items():
            if value == k or value == v:
                pos = sent.find(k)
                if pos >= 0:
                    return pos

                pos = sent.find(v)
                if pos >= 0:
                    return pos

    return pos


# TODO this one is tough, it can be easy to spot, like cheap, or it can be hard like "for upper class people" which implies high...
# TODO maybe we can use synonyms to catch most cases, eitherway - 70 instances
def priceRangeSlot(sent, value):
    """
        :param sent: target utterance
        :param value: slot value
        :return:
            Note it's only possible to have 'cheap', 'moderate', 'high', 'less than £20', '£20-25', 'more than £30'
             as possible values...
    """

    # check for the presence of the pound symbol
    pos = sent.find('\xa3')
    if pos >= 0:
        return pos

    nums_in_slot_val = re.findall('\d', value)
    if nums_in_slot_val:
        match = re.search('\d (?:pounds|dollars|euros)', sent)
        if match:
            return match.start()
        else:
            for num in nums_in_slot_val:
                pos = sent.find(num)
                if pos >= 0:
                    return pos

    for term in ['price', 'cheap', 'pricing', 'expensive', 'cost', 'affordable', 'high end']:
        pos = sent.find(term)
        if pos >= 0:
            return pos

    return pos


# TODO 36 instances in training which have the slot, but no value in the utterance
# TODO verify this is true... can we just delete this arg?
def eatTypeSlot(sent, value):
    pos = sent.find(value)
    if pos >= 0:
        return pos
    else:
        value = value.split(' ')[0]
        pos = sent.find(value)
        if pos >= 0:
            return pos

    return pos


# TODO this one is a little wierd... a highly rated restraunt could add something as simple as "great service" which is hard to detect
# TODO should we perhaps this is a case in which we just have to accept the noise, i.e. keep the arg - 185 instances
def customerRatingSlot(sent, value):
    pos = -1
    
    for term in ['customer', 'rate', 'rating', 'review']:
        pos = sent.find(term)
        if pos >= 0:
            return pos
    
    sent = re.sub('-', ' ', sent)
    for rating in ['one star', 'two star', 'three star', 'four star', 'five star']:
        pos = sent.find(rating)
        if pos >= 0:
            return pos

    match = re.search('\d star', sent)
    if match:
        return match.start()

    # sent_tokens = word_tokenize(sent)
    # for rating in ['one', 'two', 'three', 'four', 'five', 'star']:
    #     if rating in sent_tokens:
    #         return True

    return pos


# TODO @near 2 acceptable failures

# TODO @food has 24 failures which are acceptable to remove the slot
def foodSlot(sent, value):
    value = value.lower()
    sent = re.sub('-', ' ', sent)

    pos = sent.find(value)
    if pos >= 0:
        return pos
    elif value == 'english':
        return sent.find('british')
    elif value == 'fast food':
        return sent.find('american style')
    else:
        tokens = word_tokenize(sent)
        for token in tokens:
            # FIXME warning this will be slow on start up
            synsets = wordnet.synsets(token, pos='n')
            for synset in synsets:
                hypernyms = synset.hypernyms()
                while len(hypernyms) > 0:
                    lemmas = [l.name() for l in hypernyms[0].lemmas()]
                    if 'food' in lemmas:
                        return sent.find(token)
                    hypernyms = hypernyms[0].hypernyms()

    return pos


def typeSlot(sent, value):
    if value == "television" and "tv" in sent:
        return True

    return False


def hasusbportSlot(sent, value):
    curr = re.sub("-", " ", sent)
    curr = re.sub("'", "", curr)
    curr = curr.lower()
    curr_tokens = word_tokenize(curr)
    if 'usb' in curr:
        if value == "false":
            for x in ["no", "not", "non", "isnt", "arent", "dont", "doesnt", "lacking", "lacks", "without", "none", "zero", "excluded"]:
                if x in curr_tokens:
                    return True
        else:
            return True
    
    return False


def screensizeSlot(sent, value):
    value = value.split(" ")[0]
    if value in sent:
        return True

    return False


def priceSlot(sent, value):
    value = value.split(" ")[0]
    if value in sent:
        return True

    return False


def powerconsumptionSlot(sent, value):
    value = value.split(" ")[0]
    if value in sent:
        return True

    return False


def colorSlot(sent, value):
    for w in value.split(" "):
        if w not in sent and w not in [",", "and", "with"]:
            return False

    return True


def accessoriesSlot(sent, value):
    for w in value.split(" "):
        if w not in sent and w not in [",", "and", "with"]:
            return False

    return True


def weightSlot(sent, value):
    value = value.split(" ")[0]
    if value in sent:
        return True

    return False


def batterySlot(sent, value):
    value = value.split(" ")[0]
    if value in sent:
        return True

    return False


def driveSlot(sent, value):
    value = value.split(" ")[0]
    if value in sent:
        return True

    return False


def dimensionSlot(sent, value):
    value = value.split(" ")[0]
    if value in sent:
        return True

    return False


def designSlot(sent, value):
    for w in value.split(" "):
        if w not in sent and w not in [",", "and", "with"]:
            return False

    return True


def utilitySlot(sent, value):
    for w in value.split(" "):
        if w in sent and w not in [",", "and"]:
            return True

    return False


def isforbusinesscomputingSlot(sent, value):
    curr = re.sub("-", " ", sent)
    curr = re.sub("'", "", curr)
    curr = curr.lower()
    curr_tokens = word_tokenize(curr)
    if value == "false":
        if "business" in curr_tokens:
            for x in ["no", "not", "non", "isnt", "arent", "dont", "doesnt", "cant", "cannot", "except", "neither", "none"]:
                if x in curr_tokens:
                    return True
        else:
            for x in ["personal", "general", "home", "nonbusiness"]:
                if x in curr_tokens:
                    return True
    else:
        if "business" in curr:
            return True
    
    return False


def dontcareRealization(sent, slot, value):
    curr = re.sub("-", " ", sent)
    curr = re.sub("'", "", curr)
    curr = curr.lower()
    curr_tokens = word_tokenize(curr)

    if reduceSlotName(slot) in curr_tokens:
        for x in ["any", "all", "vary", "varying", "varied", "various", "variety", "different",
                  "unspecified", "irrelevant", "unnecessary", "unknown", "n/a", "particular", "specific", "priority", "choosy", "picky",
                  "regardless", "disregarding", "disregard", "excluding", "unconcerned", "matter", "specification",
                  "concern", "consideration", "considerations", "factoring", "accounting", "ignoring"]:
            if x in curr_tokens:
                return True
        for x in ["no preference", "no predetermined", "no certain", "wide range", "may or may not",
                  "not an issue", "not a factor", "not important", "not considered", "not considering", "not concerned",
                  "without a preference", "without preference", "without specification", "without caring", "without considering",
                  "not have a preference", "dont have a preference", "not consider", "dont consider", "not mind", "dont mind",
                  "not caring", "not care", "dont care", "didnt care"]:
            if x in curr:
                return True
        if ("preference" in curr_tokens or "specifics" in curr_tokens) and ("no" in curr_tokens):
            return True
    
    return False


def noneRealization(sent, slot, value):
    curr = re.sub("-", " ", sent)
    curr = re.sub("'", "", curr)
    curr = curr.lower()
    curr_tokens = word_tokenize(curr)
        
    if reduceSlotName(slot) in curr_tokens:
        for x in ["information", "info", "inform", "results", "requirement", "requirements", "specification", "specifications"]:
            if x in curr_tokens and ("no" in curr_tokens or "not" in curr_tokens):
                return True
    
    return False


def reduceSlotName(slot):
    slot = slot.replace('range', '')
    slot = slot.replace('rating', '')
    slot = slot.replace('size', '')

    if slot == 'hasusbport':
        slot == 'usb'
    elif slot == 'hdmiport':
        slot == 'hdmi'
    elif slot == 'powerconsumption':
        slot == 'power'
    elif slot == 'isforbusinesscomputing':
        slot == 'business'

    return slot.lower()



def splitContent(old_mrs, old_utterances, filename, use_heuristics=True, permute=False):
    """
    :param mr: list of dicts
    :param utterance: list
    :return:
        Splits the MRs into many MRs with individual sentences... currently this is not as robust as it could be
    """
    # delex_slots = ['name', 'eatType', 'food', 'priceRange', 'customer rating', 'area', 'familyFriendly', 'near']
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
                    elif slot_root == 'eatType':
                        slot_pos = eatTypeSlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True
                    elif slot_root == 'food':
                        slot_pos = foodSlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True
                    elif slot_root == 'priceRange':
                        slot_pos = priceRangeSlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True
                    elif slot_root == 'customer_rating':
                        slot_pos = customerRatingSlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True
                    elif slot_root == 'area':
                        slot_pos = areaSlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True
                    elif slot_root == 'familyFriendly':
                        slot_pos = familyFriendlySlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True

                    # TV dataset slots
                    elif slot_root == 'type':
                        if typeSlot(sent, value):
                            found_slot = True
                    elif slot_root == 'hasusbport':
                        if hasusbportSlot(sent, value):
                            found_slot = True
                    elif slot_root == 'screensize':
                        if screensizeSlot(sent, value):
                            found_slot = True
                    elif slot_root == 'price':
                        if priceSlot(sent, value):
                            found_slot = True
                    elif slot_root == 'powerconsumption':
                        if powerconsumptionSlot(sent, value):
                            found_slot = True
                    elif slot_root == 'color':
                        if colorSlot(sent, value):
                            found_slot = True
                    elif slot_root == 'accessories':
                        if accessoriesSlot(sent, value):
                            found_slot = True

                    # Laptop dataset slots
                    elif slot_root == 'weight':
                        if weightSlot(sent, value):
                            found_slot = True
                    elif slot_root == 'battery':
                        if batterySlot(sent, value):
                            found_slot = True
                    elif slot_root == 'drive':
                        if driveSlot(sent, value):
                            found_slot = True
                    elif slot_root == 'dimension':
                        if dimensionSlot(sent, value):
                            found_slot = True
                    elif slot_root == 'design':
                        if designSlot(sent, value):
                            found_slot = True
                    elif slot_root == 'utility':
                        if utilitySlot(sent, value):
                            found_slot = True
                    elif slot_root == 'isforbusinesscomputing':
                        if isforbusinesscomputingSlot(sent, value):
                            found_slot = True

                if found_slot:
                    new_slots[slot] = value
                    slots_found.add(slot)
                    has_slot = True

            if not has_slot:
                # if slot in ['eatType', 'familyFriendly', 'area', 'near', 'food']:
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
    with io.open(os.path.join(os.getcwd(), 'data', '_logs', filename), 'w', encoding='utf8') as log_file:
        log_file.write('\n'.join(misses))

    return new_mrs, new_utterances


def poolSlotVals(old_mrs, slots_to_pool=None):
    """
    :param old_mrs: list of mrs
    :param slots_to_pool: default (small space) = ['area', 'customer_rating', 'eatType', 'priceRange']
    :return:
    """
    # delex_slots = ['name', 'near', 'food', 'area', 'customer rating', 'familyFriendly', 'eatType', 'priceRange']
    slots = OrderedDict()
    if slots_to_pool is None:
        slots_to_pool = ['area', 'customer_rating', 'eatType', 'priceRange']
    for curr_mr in old_mrs:
        for slot, value in curr_mr.items():
            if slot in slots_to_pool:
                if slot not in slots:
                    slots[slot] = set()
                slots[slot].add(value)
    return slots


def mergeOrderedDicts(mrs, order=None):
    if order is None:
        order = ["da", "name", "eatType", "food", "priceRange", "customer_rating", "area", "familyFriendly", "near",
                 "type", "family", "hasusbport", "hdmiport", "ecorating", "screensizerange", "screensize", "pricerange", "price", "audio", "resolution", "powerconsumption", "color", "accessories", "count",
                 "processor", "memory", "driverange", "drive", "batteryrating", "battery", "weightrange", "weight", "dimension", "design", "utility", "platform", "isforbusinesscomputing", "warranty"]
    merged_mr = OrderedDict()
    for slot in order:
        for mr in mrs:
            if slot in mr:
                merged_mr[slot] = mr[slot]
                break
    return merged_mr


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


def scoreAlignment(curr_utterance, curr_mr, scoring="default+over-class"):
    '''Score a delexed utterance based on the rate of unrealized and/or overgenerated slots.
    '''
    slots_found = set()
    sent = curr_utterance
    matches = set(re.findall(r'&slot_.*?&', sent))
    num_slot_overgens = 0

    for slot, value in curr_mr.items():
        slot_root = slot.rstrip(string.digits)
        found_slot = False
        
        if slot_root == 'da':
            found_slot = True
        elif re.match(r'<.*>', slot_root):
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
                    if value_cnt > 1:
                        num_slot_overgens += value_cnt - 1
                    found_slot = True
                elif value == "dontcare":
                    if dontcareRealization(sent, slot_root, value):
                        slot_cnt = sent.count(reduceSlotName(slot_root))
                        if slot_cnt > 1:
                            num_slot_overgens += slot_cnt - 1
                        found_slot = True
                elif value == "none":
                    if noneRealization(sent, slot_root, value):
                        slot_cnt = sent.count(reduceSlotName(slot_root))
                        if slot_cnt > 1:
                            num_slot_overgens += slot_cnt - 1
                        found_slot = True
                elif slot_root == "name":
                    for pronoun in ["it", "its", "it's", "they"]:
                        if pronoun in word_tokenize(curr_utterance.lower()):
                            found_slot = True
                elif slot_root == "priceRange":
                    # if scorePriceRangeNaive(sent, value):
                    if priceRangeSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "familyFriendly":
                    if familyFriendlySlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "food":
                    if foodSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "area":
                    if areaSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "eatType":
                    if eatTypeSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "customer_rating":
                    if scoreCustomerRatingNaive(sent, value) >= 0:
                        found_slot = True
                
                elif slot_root == "type":
                    if typeSlot(sent, value):
                        found_slot = True
                elif slot_root == "hasusbport":
                    if hasusbportSlot(sent, value):
                        found_slot = True
                elif slot_root == "screensize":
                    if screensizeSlot(sent, value):
                        found_slot = True
                elif slot_root == "price":
                    if priceSlot(sent, value):
                        found_slot = True
                elif slot_root == "powerconsumption":
                    if powerconsumptionSlot(sent, value):
                        found_slot = True
                elif slot_root == "color":
                    if colorSlot(sent, value):
                        found_slot = True
                elif slot_root == "accessories":
                    if accessoriesSlot(sent, value):
                        found_slot = True

                elif slot_root == "weight":
                    if weightSlot(sent, value):
                        found_slot = True
                elif slot_root == "battery":
                    if batterySlot(sent, value):
                        found_slot = True
                elif slot_root == "drive":
                    if driveSlot(sent, value):
                        found_slot = True
                elif slot_root == "dimension":
                    if dimensionSlot(sent, value):
                        found_slot = True
                elif slot_root == "design":
                    if designSlot(sent, value):
                        found_slot = True
                elif slot_root == "utility":
                    if utilitySlot(sent, value):
                        found_slot = True
                elif slot_root == "isforbusinesscomputing":
                    if isforbusinesscomputingSlot(sent, value):
                        found_slot = True

        if found_slot:
            slots_found.add(slot)

    #if scoring == "default":
    #    return len(slots_found) / len(curr_mr)
    #elif scoring == "default+over-class":
    #    return (len(slots_found) / len(curr_mr)) / (len(matches) + 1)

    #if scoring == "default":
    #    return len(slots_found) / len(curr_mr)
    #elif scoring == "default+over-class":
    #    return (len(slots_found) - len(matches) + 1) / (len(curr_mr) + 1)

    if scoring == "default":
        return 1 / (len(curr_mr) - len(slots_found) + num_slot_overgens + 1)
    elif scoring == "default+over-class":
        return 1 / (len(curr_mr) - len(slots_found) + num_slot_overgens + 1) / (len(matches) + 1)


def count_errors(curr_utterance, curr_mr):
    '''Count unrealized and overgenerated slots in a lexicalized utterance.
    '''

    non_categorical_slots = ['familyFriendly', 'priceRange', 'customer_rating']

    slots_found = set()
    sent = curr_utterance
    matches = set(re.findall(r'&slot_.*?&', sent))
    num_slot_overgens = 0

    for slot, value in curr_mr.items():
        slot_root = slot.rstrip(string.digits)
        found_slot = False

        if slot_root == 'da':
            found_slot = True
        elif re.match(r'<.*>', slot_root):
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
                        num_slot_overgens += value_cnt - 1
                    found_slot = True
                elif value == "dontcare":
                    if dontcareRealization(sent, slot_root, value):
                        slot_cnt = sent.count(reduceSlotName(slot_root))
                        if slot_cnt > 1:
                            num_slot_overgens += slot_cnt - 1
                        found_slot = True
                elif value == "none":
                    if noneRealization(sent, slot_root, value):
                        slot_cnt = sent.count(reduceSlotName(slot_root))
                        if slot_cnt > 1:
                            num_slot_overgens += slot_cnt - 1
                        found_slot = True
                # elif slot_root == "name":
                #     for pronoun in ["it", "its", "it's", "they"]:
                #         if pronoun in word_tokenize(curr_utterance.lower()):
                #             found_slot = True
                elif slot_root == "priceRange":
                    # if scorePriceRangeNaive(sent, value):
                    if priceRangeSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "familyFriendly":
                    if familyFriendlySlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "food":
                    if foodSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "area":
                    if areaSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "eatType":
                    if eatTypeSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "customer_rating":
                    if scoreCustomerRatingNaive(sent, value) >= 0:
                        found_slot = True

                # elif slot_root == "type":
                #     if typeSlot(sent, value):
                #         found_slot = True
                # elif slot_root == "hasusbport":
                #     if hasusbportSlot(sent, value):
                #         found_slot = True
                # elif slot_root == "screensize":
                #     if screensizeSlot(sent, value):
                #         found_slot = True
                # elif slot_root == "price":
                #     if priceSlot(sent, value):
                #         found_slot = True
                # elif slot_root == "powerconsumption":
                #     if powerconsumptionSlot(sent, value):
                #         found_slot = True
                # elif slot_root == "color":
                #     if colorSlot(sent, value):
                #         found_slot = True
                # elif slot_root == "accessories":
                #     if accessoriesSlot(sent, value):
                #         found_slot = True
                #
                # elif slot_root == "weight":
                #     if weightSlot(sent, value):
                #         found_slot = True
                # elif slot_root == "battery":
                #     if batterySlot(sent, value):
                #         found_slot = True
                # elif slot_root == "drive":
                #     if driveSlot(sent, value):
                #         found_slot = True
                # elif slot_root == "dimension":
                #     if dimensionSlot(sent, value):
                #         found_slot = True
                # elif slot_root == "design":
                #     if designSlot(sent, value):
                #         found_slot = True
                # elif slot_root == "utility":
                #     if utilitySlot(sent, value):
                #         found_slot = True
                # elif slot_root == "isforbusinesscomputing":
                #     if isforbusinesscomputingSlot(sent, value):
                #         found_slot = True

        if found_slot:
            slots_found.add(slot)

    # DEBUG PRINT
    for slot in curr_mr:
        if slot not in slots_found:
            print(slot, end=' ')

    return (len(curr_mr) - len(slots_found)) + num_slot_overgens


def find_alignment(utt, mr):
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

        elif slot_root == 'eatType':
            slot_pos = eatTypeSlot(utt, value)
        elif slot_root == 'food':
            slot_pos = foodSlot(utt, value)
        elif slot_root == 'priceRange':
            slot_pos = priceRangeSlot(utt, value)
        elif slot_root == 'customer_rating':
            slot_pos = customerRatingSlot(utt, value)
        elif slot_root == 'area':
            slot_pos = areaSlot(utt, value)
        elif slot_root == 'familyFriendly':
            slot_pos = familyFriendlySlot(utt, value)

        # elif slot_root == 'type':
        #     if typeSlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'hasusbport':
        #     if hasusbportSlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'screensize':
        #     if screensizeSlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'price':
        #     if priceSlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'powerconsumption':
        #     if powerconsumptionSlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'color':
        #     if colorSlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'accessories':
        #     if accessoriesSlot(utt, value):
        #         found_slot = True
        #
        # elif slot_root == 'weight':
        #     if weightSlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'battery':
        #     if batterySlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'drive':
        #     if driveSlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'dimension':
        #     if dimensionSlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'design':
        #     if designSlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'utility':
        #     if utilitySlot(utt, value):
        #         found_slot = True
        # elif slot_root == 'isforbusinesscomputing':
        #     if isforbusinesscomputingSlot(utt, value):
        #         found_slot = True

        if slot_pos >= 0:
            alignment.append((slot_pos, slot, value))

    # sort the slot realizations by their position
    alignment.sort(key=lambda x: x[0])

    return alignment


def checkDelexSlots(slot, matches):
    for match in matches:
        if slot in match:
            return match

    return None


def extract_city(user_input, input_tokens, named_entities):
    city = None

    for ne in named_entities:
        if ne[0] == 'City':
            city = ne[2]
            break

    return city


def extract_eat_type(user_input):
    bar_synonyms = ['bar', 'bistro', 'brasserie', 'inn', 'tavern']
    coffee_shop_synonyms = ['café', 'cafe', 'coffee shop', 'coffeehouse', 'teahouse']
    restaurant_synonyms = ['cafeteria', 'canteen', 'chophouse', 'coffee shop', 'diner', 'donut shop', 'drive-in',
                           'eatery', 'eating place', 'fast-food place', 'joint', 'pizzeria', 'place to eat',
                           'restaurant', 'steakhouse']

    if any(x in user_input for x in bar_synonyms):
        return 'bar'
    elif any(x in user_input for x in coffee_shop_synonyms):
        return 'coffee shop'
    elif any(x in user_input for x in restaurant_synonyms):
        return 'restaurant'
    else:
        return None


def extract_categories(user_input, input_tokens):
    file_categories_restaurants = 'dialogue/dialogue_modules/slug2slug/data/yelp/categories_restaurants.json'

    with open(file_categories_restaurants, 'r') as f_categories:
        categories = json.load(f_categories)

        for i, token in enumerate(input_tokens):
            # search for single-word occurrences in the category list
            if token in categories:
                return {'title': token,
                        'ids': categories[token]}

            # search for bigram occurrences in the category list
            if i > 0:
                key = ' '.join(input_tokens[i-1:i+1])
                if key in categories:
                    return {'title': key,
                            'ids': categories[key]}

    return {'title': None,
            'ids': []}


def extract_price_range(user_input, input_tokens):
    CHEAP = ['1', '2']
    MODERATE = ['2', '3']
    HIGH = ['3', '4']

    indicators_indep = {'cheap': CHEAP,
                        'inexpensive': CHEAP,
                        'affordable': CHEAP,
                        'modest': CHEAP,
                        'budget': CHEAP,
                        'economic': CHEAP,
                        'economical': CHEAP,
                        'expensive': HIGH,
                        'costly': HIGH,
                        'fancy': HIGH,
                        'posh': HIGH,
                        'stylish': HIGH,
                        'elegant': HIGH,
                        'extravagant': HIGH,
                        'luxury': HIGH,
                        'luxurious': HIGH}

    indicators_indep_bigram = {'low cost': CHEAP,
                               'high class': HIGH}

    indicators_priced = {'low': CHEAP,
                         'reasonably': CHEAP,
                         'moderately': MODERATE,
                         'high': HIGH,
                         'highly': HIGH}

    indicators_range = {'low': CHEAP,
                        'moderate': MODERATE,
                        'average': MODERATE,
                        'ordinary': MODERATE,
                        'middle': MODERATE,
                        'high': HIGH}

    # search for single-word occurrences in the indicator list
    for token in input_tokens:
        if token in indicators_indep:
            return indicators_indep[token]

    # search for bigram occurrences in the category list
    for key, val in indicators_indep_bigram.items():
        if key in user_input:
            return val

    idx = -1
    try:
        idx = input_tokens.index('priced')
        if idx > 0:
            prev_token = input_tokens[idx - 1]
            if prev_token in indicators_priced:
                return indicators_priced[prev_token]
    except ValueError:
        try:
            idx = input_tokens.index('price')
        except ValueError:
            try:
                idx = input_tokens.index('prices')
            except ValueError:
                pass

        if idx > 0:
            prev_token = input_tokens[idx - 1]
            if prev_token in indicators_range:
                return indicators_range[prev_token]

    return None


def extract_area(user_input, input_tokens):
    indicators_area = ['downtown', 'city center', 'city centre', 'center of', 'centre of', 'middle of']

    area = None

    for ind in indicators_area:
        if ind in user_input:
            area = 'downtown'
            break

    return area


def extract_family_friendly(user_input, input_tokens):
    indicators = ['family', 'families', 'child', 'children', 'kid', 'kids']

    for ind in indicators:
        if ind in user_input:
            return True

    return False


def extract_near(user_input):
    indicators = ['near', 'near to', 'close to', 'next to', 'neighborhood of', 'vicinity of']

    return None


def identifySlots(user_input, named_entities):
    attributes = {}

    user_input = user_input.lower()
    input_tokens = word_tokenize(user_input)

    city = extract_city(user_input, input_tokens, named_entities)
    if city:
        attributes['city'] = city

    eat_type = extract_eat_type(user_input)
    if eat_type:
        attributes['eatType'] = eat_type

    categories = extract_categories(user_input, input_tokens)
    if categories:
        attributes['categories'] = categories

    prices = extract_price_range(user_input, input_tokens)
    if prices:
        attributes['prices'] = prices

    family_friendly = extract_family_friendly(user_input, input_tokens)
    if family_friendly:
        attributes['familyFriendly'] = family_friendly

    area = extract_area(user_input, input_tokens)
    if area:
        attributes['area'] = area

    return attributes


def testSlotOrder():
    data_frame_dev = pd.read_csv(os.path.join(os.getcwd(), "data", "devset_wrangled.csv"), header=0,
                                 encoding='utf8')  # names=['mr', 'ref']
    x_dev = data_frame_dev.mr.tolist()
    y_dev = data_frame_dev.ref.tolist()
    x_dicts = []
    for i, mr in enumerate(x_dev):
        mr_dict = OrderedDict()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            mr_dict[slot] = value
        x_dicts.append(mr_dict)
    for mr in x_dicts:
        keys = list(mr.keys())
        order = ["name", "eatType", "food", "priceRange", "customer_rating", "area", "familyFriendly", "near"]
        curr = 0
        for key in keys:
            if key in order:
                k_index = keys.index(key)
                if k_index <= order.index(key) and order.index(key) >= curr:
                    curr = order.index(key)
                else:
                    print("FAIL: %s has index %s in dev, but the order requires index %s." % (
                    key, k_index, order.index(key)))


def testSlotPooling():
    """
    Test code to test the splitting without having the run the model
    :return:
    """
    data_frame_dev = pd.read_csv(os.path.join(os.getcwd(), "data", "devset.csv"), header=0,
                                 encoding='utf8')  # names=['mr', 'ref']
    x_dev = data_frame_dev.mr.tolist()
    y_dev = data_frame_dev.ref.tolist()
    x_dicts = []
    for i, mr in enumerate(x_dev):
        mr_dict = OrderedDict()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            mr_dict[slot] = value
        x_dicts.append(mr_dict)
    slots = poolSlotVals(x_dicts)

    for slot, values in slots.items():
        print("Slot " + slot + " can have the following values: " + str(values), end="\n----\n")


def testSplitContent():
    """
    Test code to test the splitting without having the run the model
    :return:
    """
    data_frame_dev = pd.read_csv(os.path.join(os.getcwd(), "data", "devset.csv"), header=0,
                                 encoding='utf8')  # names=['mr', 'ref']
    x_dev = data_frame_dev.mr.tolist()
    y_dev = data_frame_dev.ref.tolist()
    x_dicts = []
    for i, mr in enumerate(x_dev):
        mr_dict = OrderedDict()
        if len(mr) == 0:
            continue
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            mr_dict[slot] = value
        x_dicts.append(mr_dict)
    new_x, new_y = splitContent(x_dicts, y_dev, "devset.csv")
    print("Split contents test passed.")
    # for x in range(0, len(new_x)):
    #     print(str(new_y[x]))
    #     print(str(new_x[x]))
    #     print("\n")
    # print(str(new_x))
    # print(str(new_y))


def testPermute():
    """
    Testing the permute function
    :return:
    """
    newPairs = {"There is a pizza place named Chucky Cheese.": {"name": "Chucky Cheese"},
                "Chucky Cheese Sucks.": {"name": "Chucky Cheese"},
                "It has a ball pit.": {"b": 1}, "The mascot is a giant mouse.": {"a": 1}}
    mrs, utters = permuteSentCombos(newPairs, [], [])
    for mr, utter in zip(mrs, utters):
        print(utter + " --- " + str(mr))


def wrangleSlots(filename, add_sequence_tokens=True):
    print("Aligning " + str(filename))
    data_frame_dev = pd.read_csv(os.path.join(os.getcwd(), "data", filename), header=0,
                                 encoding='utf8')  # names=['mr', 'ref']
    x_dev = data_frame_dev.mr.tolist()
    y_dev = data_frame_dev.ref.tolist()
    x_dicts = []
    for i, mr in enumerate(x_dev):
        mr_dict = OrderedDict()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            mr_dict[slot] = value
        x_dicts.append(mr_dict)
    new_x, new_y = splitContent(x_dicts, y_dev, filename.split('/')[-1], permute=False)
    filename = filename.split(".")[0] + "_wrangled.csv"
    new_file = io.open(os.path.join(os.getcwd(), "data", filename), "w", encoding='utf8')
    new_file.write("mr,ref\n")
    for row in range(0, len(new_x)):
        utterance = new_y[row]
        mr = new_x[row]
        if len(mr) == 0:
            continue
        mr_str = '"' + ', '.join(['%s[%s]' % (key, value) for (key, value) in mr.items()]) + '"'
        new_file.write(mr_str)
        new_file.write(",\"")
        new_file.write(utterance)
        new_file.write("\"\n")


def wrangleSlotsJSON(filename, add_sequence_tokens=True):
    slot_sep = ';'
    val_sep = '='
    val_sep_closing = False

    print('Aligning ' + str(filename))
    with io.open(os.path.join(os.getcwd(), 'data', filename), encoding='utf8') as f_trainset:
        # remove the comment at the beginning of the file
        for i in range(5):
            f_trainset.readline()

        # read the training data from file
        df_train = pd.read_json(f_trainset, encoding='utf8')

    x_train = df_train.iloc[:, 0].tolist()
    y_train = df_train.iloc[:, 1].tolist()

    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_train):
        x_train[i] = data_loader.preprocess_mr(mr, '(', slot_sep, val_sep)
        
    x_dicts = []
    for i, mr in enumerate(x_train):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value
        x_dicts.append(mr_dict)

    new_x, new_y = splitContent(x_dicts, y_train, filename.split('/')[-1], permute=False)

    data_new = []
    filename = filename.split('.')[0] + '_wrangled.json'
    for row in range(0, len(new_x)):
        utterance = new_y[row]
        mr = new_x[row]
        if len(mr) == 0:
            continue
        mr_str = mr.pop('da')
        mr_str += '(' + ';'.join(['%s=%s' % (key.rstrip(string.digits), value) for (key, value) in mr.items()]) + ')'

        data_new.append([])
        data_new[row].extend([mr_str, utterance])

    with io.open(os.path.join(os.getcwd(), 'data', filename), 'w', encoding='utf8') as f_data_new:
        json.dump(data_new, f_data_new, indent=4)


def align_slots(file_path):
    print('Aligning ' + str(file_path))

    df_dataset = pd.read_csv(file_path, header=0, encoding='utf8')
    x = df_dataset.mr.tolist()
    y = df_dataset.ref.tolist()

    alignments = []
    for i, mr in enumerate(x):
        mr_dict = OrderedDict()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            mr_dict[slot] = value

        alignments.append(find_alignment(y[i], mr_dict))

    alignment_strings = []
    for i in range(len(y)):
        alignment_strings.append(' '.join(['({0}: {1})'.format(pos, slot) for pos, slot, _ in alignments[i]]))

    new_df = pd.DataFrame(columns=['mr', 'ref', 'alignment'])
    new_df['mr'] = x
    new_df['ref'] = y
    new_df['alignment'] = alignment_strings

    new_df.to_csv(file_path.replace('.csv', '_aligned.csv'), index=False, encoding='utf8')


def score_slot_realizations(file_path):
    print('Analyzing missing/extra slot realizations ' + str(file_path))

    slot_sep = ','
    val_sep = '['
    val_sep_closing = True

    # slot_cnt = 0

    df_dataset = pd.read_csv(file_path, header=0, encoding='utf8')
    x = df_dataset.iloc[:, 0].tolist()
    y = df_dataset.iloc[:, 1].tolist()
    y = [data_loader.preprocess_utterance(utt) for utt in y]

    misses = []
    for i, mr in enumerate(x):
        mr_dict = OrderedDict()

        # extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value.lower()

            # if not re.match(r'<.*>', slot):
            #     slot_cnt += 1

        # TODO: get rid of this hack
        # move the food-slot to the end of the dict (because of delexing)
        if 'food' in mr_dict:
            food_val = mr_dict['food']
            del(mr_dict['food'])
            mr_dict['food'] = food_val

        # delexicalize the MR and the utterance
        y[i] = ' '.join(data_loader.delex_sample(mr_dict, y[i]))

        print(str(i) + '.\t', end='')
        misses.append(count_errors(y[i], mr_dict))
        print()

    # DEBUG PRINT
    # print(slot_cnt)

    new_df = pd.DataFrame(columns=['mr', 'ref', 'misses'])
    new_df['mr'] = x
    new_df['ref'] = y
    new_df['misses'] = misses

    new_df.to_csv(file_path.replace('.csv', '_misses.csv'), index=False, encoding='utf8')


def augment_with_emphasis(file_path):
    print('Augmenting MRs with emphasis in ' + str(file_path))

    df_dataset = pd.read_csv(file_path, header=0, encoding='utf8')
    x = df_dataset.mr.tolist()
    y = df_dataset.ref.tolist()

    alignments = []
    for i, mr in enumerate(x):
        mr_dict = OrderedDict()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            mr_dict[slot] = value

        alignments.append(find_alignment(y[i], mr_dict))

    for i in range(len(y)):
        for pos, slot, _ in alignments[i]:
            if slot == 'name':
                break
            x[i] = x[i].replace(slot, '<emph>[], ' + slot)

    new_df = pd.DataFrame(columns=['mr', 'ref'])
    new_df['mr'] = x
    new_df['ref'] = y

    new_df.to_csv(file_path.replace('.csv', '_augm_emph.csv'), index=False, encoding='utf8')


def evaluate_emphasis(file_path):
    '''Determines how many of the indicated emphasis instances are realized in the utterance.
    '''

    emph_token = '<emph>'

    df_dataset = pd.read_csv(file_path, header=0, encoding='utf8')
    x = df_dataset.mr.tolist()
    y = df_dataset.utterance.tolist()

    x_dicts = []
    emph_missed = []
    emph_total = []
    for i, mr in enumerate(x):
        expect_emph = False
        emph_slots = set()
        mr_dict = OrderedDict()

        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()

            # extract tokens to be emphasized
            if slot == emph_token:
                expect_emph = True
            else:
                mr_dict[slot] = value
                if expect_emph:
                    emph_slots.add(slot)
                    expect_emph = False

        alignment = find_alignment(y[i], mr_dict)

        emph_total.append(len(emph_slots))

        # check how many emphasized slots were not realized before the name-slot
        for pos, slot, _ in alignment:
            # DEBUG PRINT
            # print(alignment)
            # print(emph_slots)
            # print()

            if slot == 'name':
                break

            if slot in emph_slots:
                emph_slots.remove(slot)

        emph_missed.append(len(emph_slots))

    new_df = pd.DataFrame(columns=['mr', 'ref', 'emphasis (missed)', 'emphasis (total)'])
    new_df['mr'] = x
    new_df['ref'] = y
    new_df['emphasis (missed)'] = emph_missed
    new_df['emphasis (total)'] = emph_total

    new_df.to_csv(file_path.replace('.csv', '_eval_emph.csv'), index=False, encoding='utf8')


def augment_with_contrast(file_path):
    contrast_connectors = ['but', 'however', 'yet']
    scalar_slots = {
        'customer_rating': {
            'low': 1,
            'average': 2,
            'high': 3,
            '1 out of 5': 1,
            '3 out of 5': 2,
            '5 out of 5': 3
        },
        'priceRange': {
            'high': 1,
            'moderate': 2,
            'cheap': 3,
            'more than £30': 1,
            '£20-25': 2,
            'less than £20': 3
        },
        'familyFriendly': {
            'no': 1,
            'yes': 3
        }
    }

    print('Augmenting MRs with contrast in ' + str(file_path))

    df_dataset = pd.read_csv(file_path, header=0, encoding='utf8')
    x = df_dataset.mr.tolist()
    y = df_dataset.ref.tolist()

    alignments = []
    for i, mr in enumerate(x):
        mr_dict = OrderedDict()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            mr_dict[slot] = value

        alignments.append(find_alignment(y[i], mr_dict))

    for i in range(len(y)):
        for contrast_conn in contrast_connectors:
            contrast_pos = y[i].find(contrast_conn)
            if contrast_pos >= 0:
                slot_before = None
                value_before = None
                slot_after = None
                value_after = None

                for pos, slot, value in alignments[i]:
                    if pos > contrast_pos:
                        if not slot_before:
                            break
                        if slot in scalar_slots:
                            slot_after = slot
                            value_after = value
                            break
                    else:
                        if slot in scalar_slots:
                            slot_before = slot
                            value_before = value

                if slot_before and slot_after:
                    if slot_before in scalar_slots and slot_after in scalar_slots:
                        if scalar_slots[slot_before][value_before] - scalar_slots[slot_after][value_after] == 0:
                            x[i] += ', <concession>[{0} {1}]'.format(slot_before, slot_after)
                        else:
                            x[i] += ', <contrast>[{0} {1}]'.format(slot_before, slot_after)

                break

    new_df = pd.DataFrame(columns=['mr', 'ref'])
    new_df['mr'] = x
    new_df['ref'] = y

    new_df.to_csv(file_path.replace('.csv', '_augm_contrast.csv'), index=False, encoding='utf8')


def augment_with_contrast_tgen(file_path):
    contrast_connectors = ['but', 'however', 'yet']
    scalar_slots = {
        'customer_rating': {
            'low': 1,
            'average': 2,
            'high': 3,
            '1 out of 5': 1,
            '3 out of 5': 2,
            '5 out of 5': 3
        },
        'priceRange': {
            'high': 1,
            'moderate': 2,
            'cheap': 3,
            'more than £30': 1,
            '£20-25': 2,
            'less than £20': 3
        },
        'familyFriendly': {
            'no': 1,
            'yes': 3
        }
    }

    print('Augmenting MRs with contrast in ' + str(file_path))

    df_dataset = pd.read_csv(file_path, header=0, encoding='utf8')
    x = df_dataset.mr.tolist()
    y = df_dataset.ref.tolist()

    alignments = []
    for i, mr in enumerate(x):
        mr_dict = OrderedDict()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            mr_dict[slot] = value

        alignments.append(find_alignment(y[i], mr_dict))

    contrasts = []

    for i in range(len(y)):
        contrasts.append(['none', 'none', 'none', 'none'])
        for contrast_conn in contrast_connectors:
            contrast_pos = y[i].find(contrast_conn)
            if contrast_pos >= 0:
                slot_before = None
                value_before = None
                slot_after = None
                value_after = None

                for pos, slot, value in alignments[i]:
                    if pos > contrast_pos:
                        if not slot_before:
                            break
                        if slot in scalar_slots:
                            slot_after = slot
                            value_after = value
                            break
                    else:
                        if slot in scalar_slots:
                            slot_before = slot
                            value_before = value

                if slot_before and slot_after:
                    if scalar_slots[slot_before][value_before] - scalar_slots[slot_after][value_after] == 0:
                        contrasts[i][2] = slot_before
                        contrasts[i][3] = slot_after
                    else:
                        contrasts[i][0] = slot_before
                        contrasts[i][1] = slot_after
                break

    new_df = pd.DataFrame(columns=['mr', 'ref', 'contrast1', 'contrast2', 'concession1', 'concession2'])
    new_df['mr'] = x
    new_df['ref'] = y
    new_df['contrast1'] = [tup[0] for tup in contrasts]
    new_df['contrast2'] = [tup[1] for tup in contrasts]
    new_df['concession1'] = [tup[2] for tup in contrasts]
    new_df['concession2'] = [tup[3] for tup in contrasts]

    new_df.to_csv(file_path.replace('.csv', '_augm_contrast_tgen.csv'), index=False, encoding='utf8')


if __name__ == '__main__':
    # wrangleSlots('data/rest_e2e/trainset_e2e.csv')
    # align_slots('data/rest_e2e/trainset_e2e.csv')
    # augment_with_emphasis('data/rest_e2e/trainset_e2e.csv')
    # augment_with_contrast('data/rest_e2e/trainset_stylistic_thresh_2_augm_5.csv')
    # augment_with_contrast_tgen('data/rest_e2e/devset_e2e.csv')
    # evaluate_emphasis('eval/predictions-rest_e2e_stylistic_selection/devset/predictions RNN (4+4) (16k) utt split.csv')
    score_slot_realizations('eval/predictions-rest_e2e/devset/predictions_devset_ensemble.csv')

    # user_input = 'Is there a family-friendly bar in downtown santa cruz that serves reasonably priced burgers?'
    # gnode_entities = [('VisualArtwork', 282.797767, 'restaurant in'), ('City', 2522.766114, 'Santa Cruz')]
    # print(identifySlots(user_input, gnode_entities))

    #wrangleSlotsJSON('data/tv/train.json')
    # wrangleSlotsJSON('data/laptop/train.json')

    # testSlotOrder()

    # foodSlot('This is a test of pasta', 'English')
    # testPermute()
    # testSplitContent()
    # testSlotPooling()
