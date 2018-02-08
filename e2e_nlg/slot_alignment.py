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
    # children-friendly, children friendly, family-friendly, family friendly, kid-friendly, kid friendly
    curr = re.sub("-", " ", sent)
    curr = re.sub("'", "", curr)
    curr = curr.lower()
    curr_tokens = word_tokenize(curr.lower())
    for sval in ["famil", "kid", "child"]:
        if sval in curr:
            # root_sval = sval.split(" ")
            # curr_index = -1
            # neg_indicies = -1
            # for x, tok in enumerate(curr_tokens):
            #     if sval in tok:
            # if root_sval[0] == tok:
            # break
            # elif tok in ["not", "non"]:
            #     neg_indicies = x
            # if (value == "no" and neg_indicies != -1 and curr_index != -1) or value == "yes":
            # sval = " ".join(curr_tokens[neg_indicies:curr_index+2])
            if value == "no":
                for x in ["not", "non", "isnt", "dont", "doesnt", "lowly", "lacking", "bad", "none"]:
                    if x in curr_tokens:
                        return True
            else:
                return True
        elif value == "no" and "adult" in sent:
            return True
    # print(curr)
    return False


# TODO 166 instances in training which have the slot, but no value in the utterance or worse, city center is tn utterance
# TODO verify this is true... can we just delete this arg? It seems maybe in rl city center is near the river, hence they are interchangable?
def areaSlot(sent, value):
    """
    :param sent: target utterance
    :param value: slot value
    :return:
        Note it's only possible to have city centre and riverside as possible values...
    """
    if value == "riverside" and "river" in sent:
        return True
    elif value == "city centre":
        if "center" in sent or "centre" in sent or ("middle" in sent and "city" in sent):
            return True
    return False


def scorePriceRangeNaive(sent, value):
    pot_values = {'moderate':'£20-25', 'high':'more than £30', 'cheap':'less than £20'}
    for k,v in pot_values.items():
        if value == k or value == v:
            if k in sent or v in sent:
                return True
    return False


def scoreCustomerRatingNaive(sent, value):
    pot_values = {'1 out of 5':'low', '3 out of 5':'average', '5 out of 5':'high'}
    for k,v in pot_values.items():
        if value == k or value == v:
            if k in sent:
                return True
            if v in sent:
                if v == 'high' and 'high price' in sent:
                    if 'high customer' in sent:
                        return True
                else:
                    return True
    return False

# TODO this one is tough, it can be easy to spot, like cheap, or it can be hard like "for upper class people" which implies high...
# TODO maybe we can use synonyms to catch most cases, eitherway - 70 instances
def priceRangeSlot(sent, value):
    """
        :param sent: target utterance
        :param value: slot value
        :return:
            Note it's only possible to have 'moderate', 'high', 'less than £20', 'more than £30', 'cheap', '£20-25'
             as possible values...
        """
    if value in sent or '\xa3' in sent:
        return True
    nums = re.findall("\d", value)
    if nums:
        if re.findall("\d pounds", sent):
            return True
        else:
            for num in nums:
                if num in sent:
                    return True
    for term in ["price", "cheap", "pricing", "expensive", "cost", "affordable", "high end"]:
        if term in sent:
            return True
    return False


# TODO 36 instances in training which have the slot, but no value in the utterance
# TODO verify this is true... can we just delete this arg?
def eatTypeSlot(sent, value):
    if value in sent:
        return True
    else:
        value = value.split(" ")[0]
        if value in sent:
            return True
    return False


# TODO this one is a little wierd... a highly rated restraunt could add something as simple as "great service" which is hard to detect
# TODO should we perhaps this is a case in which we just have to accept the noise, i.e. keep the arg - 185 instances
def customerRatingSlot(sent, value):
    if value in sent:
        return True
    else:
        if "rated" in sent or "rating" in sent:
            return True
        else:
            sent = re.sub("-", " ", sent)
            for rating in ["one star", "two star", "three star", "four star", "five star", "review"]:
                if rating in sent:
                    return True
            tokens = word_tokenize(sent)
            for rating in ["three", "four", "five", "one", "two", "star"]:
                if rating in tokens:
                    return True
    return False


# TODO @near 2 acceptable failures

# TODO @food has 24 failures which are acceptable to remove the slot
def foodSlot(sent, value):
    value = value.lower()
    sent = re.sub("-", " ", sent.lower())
    if value in sent:
        return True
    elif value == "english" and "british" in sent:
        return True
    elif value == "fast food" and "american style" in sent:
        return True
    else:
        tokens = word_tokenize(sent)
        for token in tokens:
            # FIXME warning this will be slow on start up
            synsets = wordnet.synsets(token, pos='n')
            for synset in synsets:
                hypernyms = synset.hypernyms()
                while len(hypernyms) > 0:
                    lemmas = [l.name() for l in hypernyms[0].lemmas()]
                    if "food" in lemmas:
                        return True
                    hypernyms = hypernyms[0].hypernyms()
    return False


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
        Splits the mrs into many mrs with smaller sentences... currently this is not as robust as it could be
    """
    # delex_slots = ['name', 'near', 'food', 'area', 'customer rating', 'familyFriendly', 'eatType', 'priceRange']
    new_mrs = []
    new_utterances = []
    slot_fails = OrderedDict()
    instance_fails = set()
    misses = ["The following samples were removed: "]
    base = max(int(len(old_utterances) * .1), 1)
    benchmarks = [base * i for i in range(1, 11)]
    for index in range(0, len(old_mrs)):
        if index in benchmarks:
            curr_state = index / base
            print("Slot alignment is " + str(10 * curr_state) + "% done.")
        curr_mr = old_mrs[index]
        curr_utterance = old_utterances[index]
        curr_utterance = re.sub(r'\s+', ' ', curr_utterance).strip()
        sents = sent_tokenize(curr_utterance)
        root_utterance = sents[0]
        new_pair = {sent: OrderedDict() for sent in sents}
        foundSlots = set()
        rm_slot = []
        for slot, value in curr_mr.items():
            slot_root = slot.rstrip(string.digits)
            hasSlot = False
            for sent, new_slots in new_pair.items():
                found_slot = False
                sent = sent.lower()
                if value.lower() in sent.lower():
                    found_slot = True
                elif slot_root == "da":
                    found_slot = True
                elif slot_root == "name":
                    for pronoun in ["it", "its", "it's", "they"]:
                        if pronoun in word_tokenize(curr_utterance.lower()):
                            found_slot = True
                elif use_heuristics:
                    if value == "dontcare":
                        if dontcareRealization(sent, slot_root, value):
                            found_slot = True
                    elif value == "none":
                        if noneRealization(sent, slot_root, value):
                            found_slot = True
                    elif slot_root == "priceRange":
                        if priceRangeSlot(sent, value):
                            found_slot = True
                    elif slot_root == "familyFriendly":
                        if familyFriendlySlot(sent, value):
                            found_slot = True
                    elif slot_root == "food":
                        if foodSlot(sent, value):
                            found_slot = True
                    elif slot_root == "area":
                        if areaSlot(sent, value):
                            found_slot = True
                    elif slot_root == "eatType":
                        if eatTypeSlot(sent, value):
                            found_slot = True
                    elif slot_root == "customer_rating":
                        if customerRatingSlot(sent, value):
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
                    new_slots[slot] = value
                    foundSlots.add(slot)
                    hasSlot = True

            if not hasSlot:
                # if slot in ["eatType", "familyFriendly", "area", "near", "food"]:
                misses.append("Couldn't find " + slot + "(" + value + ") - " + old_utterances[index])
                rm_slot.append(slot)
                # continue
                instance_fails.add(curr_utterance)
                if slot not in slot_fails:
                    slot_fails[slot] = 0
                slot_fails[slot] += 1
                # if slot == "customer_rating":
                #     print("Couldn't find " + slot + "(" + value + ")in " + old_utterances[index])
        else:
            for slot in rm_slot:
                del curr_mr[slot]
            new_mrs.append(curr_mr)
            new_utterances.append(curr_utterance.strip())
            if len(new_pair) > 1:
                for sent, new_slots in new_pair.items():
                    if root_utterance == sent:
                        new_slots["position"] = "outer"
                    else:
                        new_slots["position"] = "inner"
                    new_mrs.append(new_slots)
                    new_utterances.append(sent)
            if permute:
                permuteSentCombos(new_pair, new_mrs, new_utterances, max_iter=True)
    misses.append("We had these misses from all categories: " + str(slot_fails.items()))
    misses.append("So we had " + str(len(instance_fails)) + " samples with misses out of " + str(len(old_utterances)))
    with io.open(os.path.join(os.getcwd(), "data", "_logs", filename), 'w', encoding='utf8') as log_file:
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
    foundSlots = set()
    sent = curr_utterance
    matches = set(re.findall(r'&slot_.*?&', sent))
    slot_overgens = 0

    for slot, value in curr_mr.items():
        slot_root = slot.rstrip(string.digits)
        found_slot = False
        
        if slot_root == "da":
            found_slot = True
        else:
            delex_slot = checkDelexSlots(slot, matches)
            if delex_slot:
                found_slot = True
                matches.remove(delex_slot)
            else:
                sent = sent.lower()
                if value.lower() in sent.lower():
                    value_cnt = sent.lower().count(value.lower())
                    if value_cnt > 1:
                        slot_overgens += value_cnt - 1
                    found_slot = True
                elif value == "dontcare":
                    if dontcareRealization(sent, slot_root, value):
                        slot_cnt = sent.lower().count(reduceSlotName(slot_root))
                        if slot_cnt > 1:
                            slot_overgens += slot_cnt - 1
                        found_slot = True
                elif value == "none":
                    if noneRealization(sent, slot_root, value):
                        slot_cnt = sent.lower().count(reduceSlotName(slot_root))
                        if slot_cnt > 1:
                            slot_overgens += slot_cnt - 1
                        found_slot = True
                elif slot_root == "name":
                    for pronoun in ["it", "its", "it's", "they"]:
                        if pronoun in word_tokenize(curr_utterance.lower()):
                            found_slot = True
                elif slot_root == "priceRange":
                    if scorePriceRangeNaive(sent, value):
                    # if priceRangeSlot(sent, value):
                        found_slot = True
                elif slot_root == "familyFriendly":
                    if familyFriendlySlot(sent, value):
                        found_slot = True
                elif slot_root == "food":
                    if foodSlot(sent, value):
                        found_slot = True
                elif slot_root == "area":
                    if areaSlot(sent, value):
                        found_slot = True
                elif slot_root == "eatType":
                    if eatTypeSlot(sent, value):
                        found_slot = True
                elif slot_root == "customer_rating":
                    if scoreCustomerRatingNaive(sent, value):
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
            foundSlots.add(slot)
            continue

    #if scoring == "default":
    #    return len(foundSlots) / len(curr_mr)
    #elif scoring == "default+over-class":
    #    return (len(foundSlots) / len(curr_mr)) / (len(matches) + 1)

    #if scoring == "default":
    #    return len(foundSlots) / len(curr_mr)
    #elif scoring == "default+over-class":
    #    return (len(foundSlots) - len(matches) + 1) / (len(curr_mr) + 1)

    if scoring == "default":
        return len(curr_mr) / (len(curr_mr) - len(foundSlots) + slot_overgens + 1)
    elif scoring == "default+over-class":
        return len(curr_mr) / (len(curr_mr) - len(foundSlots) + slot_overgens + 1) / (len(matches) + 1)


def checkDelexSlots(slot, matches):
    for match in matches:
        if slot in match:
            return match
    return False


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


if __name__ == '__main__':
    #wrangleSlots('rest_e2e/trainset_e2e.csv')

    #wrangleSlotsJSON('tv/train.json')
    wrangleSlotsJSON('laptop/train.json')

    # testSlotOrder()

    # foodSlot('This is a test of pasta', 'English')
    # testPermute()
    # testSplitContent()
    # testSlotPooling()
