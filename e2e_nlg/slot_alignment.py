import sys
import os
import io
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

#TODO 139 instances in training which have the slot, but no value in the utterance
#TODO verify this is true... can we just delete this arg?
def familyFriendlySlot(sent, value):
    # children-friendly, children friendly, family-friendly, family friendly, kid-friendly, kid friendly
    curr = re.sub("-", " ", sent)
    curr = curr.lower()
    # curr_tokens = word_tokenize(curr.lower())
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
                return True
        elif value == "no" and "adult" in sent:
            return True
    # print(curr)
    return False

#TODO 166 instances in training which have the slot, but no value in the utterance or worse, city center is tn utterance
#TODO verify this is true... can we just delete this arg? It seems maybe in rl city center is near the river, hence they are interchangable?
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

#TODO this one is tough, it can be easy to spot, like cheap, or it can be hard like "for upper class people" which implies high...
#TODO maybe we can use synonyms to catch most cases, eitherway - 70 instances
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

#TODO 36 instances in training which have the slot, but no value in the utterance
#TODO verify this is true... can we just delete this arg?
def eatTypeSlot(sent, value):
    if value in sent:
        return True
    else:
        value = value.split(" ")[0]
        if value in sent:
            return True
    return False

#TODO this one is a little wierd... a highly rated restraunt could add something as simple as "great service" which is hard to detect
#TODO should we perhaps this is a case in which we just have to accept the noise, i.e. keep the arg - 185 instances
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
            for rating in ["three", "four", "five", "one", "two"]:
                if rating in tokens:
                    return True
    return False


#TODO @near 2 acceptable failures

#TODO @food has 24 failures which are acceptable to remove the slot
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
            #FIXME warning this will be slow on start up
            synsets = wordnet.synsets(token, pos='n')
            for synset in synsets:
                hypernyms = synset.hypernyms()
                while len(hypernyms) > 0:
                    lemmas = [l.name() for l in hypernyms[0].lemmas()]
                    if "food" in lemmas:
                        return True
                    hypernyms = hypernyms[0].hypernyms()
    return False


def splitContent(old_mrs, old_utterances, filename, use_heuristics=True, permute=True, remove_misses=True):
    """
    :param mr: list of dicts
    :param utterance: list
    :return:
        Splits the mrs into many mrs with smaller sentences... currently this is not as robust as it could be
    """
    # delex_slots = ['name', 'near', 'food', 'area', 'customer rating', 'familyFriendly', 'eatType', 'priceRange']
    new_mrs = []
    new_utterances = []
    slot_fails = {}
    instance_fails = set()
    misses = ["The following samples were removed: "]
    base = max(int(len(old_utterances) * .1), 1)
    benchmarks = [base * i for i in range(1, 11)]
    for index in range(0,len(old_mrs)):
        if index in benchmarks:
            curr_state = index / base
            print("Slot alignment is " + str(10 * curr_state) + "% done.")
        curr_mr = old_mrs[index]
        curr_utterance = old_utterances[index]
        new_pair = {re.sub(r'\s+', ' ', sent).strip():{} for sent in sent_tokenize(curr_utterance)}
        foundSlots = set()
        rm_slot = []
        for slot, value in curr_mr.items():
            hasSlot = False
            for sent, new_slots in new_pair.items():
                foundSlot = False
                sent = sent.lower()
                if value.lower() in sent.lower():
                    foundSlot = True
                elif slot == "name":
                    for pronoun in ["it", "its", "it's"]:
                        if pronoun in word_tokenize(curr_utterance.lower()):
                            foundSlot = True
                elif use_heuristics:
                    if slot == "priceRange":
                        if priceRangeSlot(sent, value):
                            foundSlot = True
                    elif slot == "familyFriendly":
                        if familyFriendlySlot(sent, value):
                            foundSlot = True
                    elif slot == "food":
                        if foodSlot(sent, value):
                            foundSlot = True
                    elif slot == "area":
                        if areaSlot(sent, value):
                            foundSlot = True
                    elif slot == "eatType":
                        if eatTypeSlot(sent, value):
                            foundSlot = True
                    elif slot == "customer_rating":
                        if customerRatingSlot(sent, value):
                            foundSlot = True
                if foundSlot:
                    new_slots[slot] = value
                    foundSlots.add(slot)
                    hasSlot = True
                    continue
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
                    new_mrs.append(new_slots)
                    new_utterances.append(sent)
            if permute:
                permuteSentCombos(new_pair, new_mrs, new_utterances)
    misses.append("We had these misses from all categories: " + str(slot_fails.items()))
    misses.append("So we had " + str(len(instance_fails)) + " samples with misses out of " + str(len(old_utterances)))
    with io.open(os.path.join(os.getcwd(), "data", "logs", filename), 'w', encoding='utf8') as log_file:
        log_file.write('\n'.join(misses))

    return new_mrs, new_utterances


def poolSlotVals(old_mrs, slots_to_pool=None):
    """
    :param old_mrs: list of mrs
    :param slots_to_pool: default (small space) = ['area', 'customer_rating', 'eatType', 'priceRange']
    :return:
    """
    # delex_slots = ['name', 'near', 'food', 'area', 'customer rating', 'familyFriendly', 'eatType', 'priceRange']
    slots = {}
    if slots_to_pool is None:
        slots_to_pool = ['area', 'customer_rating', 'eatType', 'priceRange']
    for curr_mr in old_mrs:
        for slot, value in curr_mr.items():
            if slot in slots_to_pool:
                if slot not in slots:
                    slots[slot] = set()
                slots[slot].add(value)
    return slots


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
                new_mr, new_utterance = mergeEntries([root]+comb)
                new_utterance = new_utterance.strip()
                if new_utterance not in utterances:
                    mrs.append(new_mr)
                    utterances.append(new_utterance)

        if assume_root:
            break
    #frivolous return for potential debug
    return utterances, mrs


def mergeEntries(merge_tuples):
    """
    :param merge_tuples: list of (utterance, mr) tuples to merge into one pair
    :return:
    """
    sent = ""
    mr = {}
    for curr_sent, curr_mr in merge_tuples:
        sent += " " + curr_sent
        mr.update(curr_mr)
    return mr, sent


def testSlotPooling():
    """
    Test code to test the splitting without having the run the model
    :return:
    """
    data_frame_dev = pd.read_csv(os.path.join(os.getcwd(), "data", "devset.csv"), header=0, encoding='utf8')  # names=['mr', 'ref']
    x_dev = data_frame_dev.mr.tolist()
    y_dev = data_frame_dev.ref.tolist()
    x_dicts = []
    for i, mr in enumerate(x_dev):
        mr_dict = {}
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
    data_frame_dev = pd.read_csv(os.path.join(os.getcwd(), "data", "devset.csv"), header=0, encoding='utf8')  # names=['mr', 'ref']
    x_dev = data_frame_dev.mr.tolist()
    y_dev = data_frame_dev.ref.tolist()
    x_dicts = []
    for i, mr in enumerate(x_dev):
        mr_dict = {}
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            mr_dict[slot] = value
        x_dicts.append(mr_dict)
    new_x, new_y = splitContent(x_dicts, y_dev)
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
    newPairs = {"There is a pizza place named Chucky Cheese.": {"name":"Chucky Cheese"}, "Chucky Cheese Sucks.": {"name":"Chucky Cheese"},
                "It has a ball pit.":{"b":1}, "The mascot is a giant mouse.":{"a":1}}
    mrs, utters = permuteSentCombos(newPairs, [], [])
    for mr, utter in zip(mrs, utters):
        print(utter + " --- " + str(mr))

def wrangleSlots(filename):
    data_frame_dev = pd.read_csv(os.path.join(os.getcwd(), "data", filename), header=0,
                                 encoding='utf8')  # names=['mr', 'ref']
    x_dev = data_frame_dev.mr.tolist()
    y_dev = data_frame_dev.ref.tolist()
    x_dicts = []
    for i, mr in enumerate(x_dev):
        mr_dict = {}
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            slot = slot.replace(' ', '_')
            # parse the value
            value = slot_value[sep_idx + 1:-1].strip()
            mr_dict[slot] = value
        x_dicts.append(mr_dict)
    new_x, new_y = splitContent(x_dicts, y_dev, filename)
    filename = filename.split(".")[0]+"_wrangled.csv"
    new_file = open(os.path.join(os.getcwd(), "data", filename), "w")
    for row in range(0, len(new_x)):
        utterance = new_y[row]
        mr = new_x[row]
        mr_str = '"'+', '.join(['%s[%s]' % (key, value) for (key, value) in mr.items()])+'"'
        new_file.write(mr_str)
        new_file.write(",")
        new_file.write(utterance)
        new_file.write("\n")

wrangleSlots("trainset.csv")

# foodSlot("This is a test of pasta", "English")
# testPermute()
# testSplitContent()
# testSlotPooling()