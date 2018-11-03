# -*- coding: utf-8 -*-

import os
import io
import string
import re
import itertools
from collections import OrderedDict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet

import config


NEG_THRESH = 10

negation_cues = [
    'no', 'not', 'non', 'none', 'nor', 'never',
    'n\'t', 'isnt', 'cant', 'cannot', 'doesnt', 'dont', 'didnt',
    'lack', 'lacks', 'lacking', 'unavailable', 'without'
]
negation_phrases = [
    'everything but', 'out of luck'
]


def find_first_in_list(val, lst):
    try:
        return lst.index(val)
    except ValueError as e:
        return -1


def find_all_in_list(val, lst):
    positions = []

    for pos, elem in enumerate(lst):
        if elem == val:
            positions.append(pos)

    return positions


def playerPerspectiveSlot(sent, value):
    leftmost_pos = -1

    sent = re.sub('-', ' ', sent)
    sent_tok = word_tokenize(sent)

    # Split the slot-value into individual items and extract the first word of each item only
    items = [item.split(' ')[0] for item in value.split('; ')]

    # Search for all individual items exhaustively
    for item in items:
        idx = find_first_in_list(item, sent_tok)
        if idx >= 0:
            pos = sent.find(item)
            if leftmost_pos == -1 or pos < leftmost_pos:
                leftmost_pos = pos
        else:
            # Alternative for when the value is 'first person'
            if value == 'first person':
                idx2 = find_first_in_list('fps', sent_tok)
                if idx2 >= 0:
                    return sent.find('fps')
            elif value == 'bird view':
                pos2 = sent.find('top down')
                if pos2 >= 0:
                    return pos2

            return -1

    return leftmost_pos


def genreSlot(sent, value):
    leftmost_pos = -1

    # Split the slot-value into individual items
    genres = [item for item in value.split('; ')]

    # Define root forms of the genre terms
    genres_root = []
    for genre in genres:
        if genre == 'action-adventure':
            genres_root.extend(['action', 'adventur'])
        elif genre == 'adventure':
            genres_root.append('adventur')
        elif genre == 'driving/racing':
            genres_root.append(['driving', 'drive', 'racing', 'race'])
        elif genre == 'fighting':
            genres_root.append('fight')
        elif genre == 'mmorpg':
            genres_root.append(['mmorpg', 'massively'])
        elif genre == 'platformer':
            genres_root.append(['platformer', 'platforming'])
        elif genre == 'real-time strategy':
            genres_root.append(['real-time', 'rts'])
        elif genre == 'role-playing':
            genres_root.append(['role-play', 'rpg'])
        elif genre == 'shooter':
            genres_root.append(['shoot', 'fps'])
        elif genre == 'simulation':
            genres_root.append(['simulat', ' sim'])
        elif genre == 'strategy':
            genres_root.append('strateg')
        elif genre == 'tactical':
            genres_root.append('tactic')
        elif genre == 'trivia/board game':
            genres_root.extend(['trivia', 'board'])
        elif genre == 'turn-based strategy':
            genres_root.append('turn-based')
        elif genre == 'vehicular combat':
            genres_root.extend(['vehic', 'combat'])
        else:
            genres_root.append(genre)

    # Search for all individual items exhaustively
    for keywords in genres_root:
        variant_found = False

        if not isinstance(keywords, list):
            keywords = [keywords]

        for kw in keywords:
            pos = sent.find(kw)
            if pos >= 0:
                variant_found = True
                if leftmost_pos == -1 or pos < leftmost_pos:
                    leftmost_pos = pos

        if not variant_found:
            return -1

    return leftmost_pos


def platformsSlot(sent, value):
    leftmost_pos = -1

    # Split the slot-value into individual items and extract the first word of each item only
    platforms = [item.split(' ')[0] for item in value.split('; ')]

    # Search for all individual items exhaustively
    for platform in platforms:
        pos = sent.find(platform)
        if pos >= 0:
            if leftmost_pos == -1 or pos < leftmost_pos:
                leftmost_pos = pos
        else:
            return pos

    return leftmost_pos


def esrbSlot(sent, value):
    pos = -1
    sent = re.sub('-', ' ', sent)

    rating_poss_vals = [value]
    if value == 'e (for everyone)':
        rating_poss_vals.extend(['e rated', 'rated e', 'e rating', 'rating e', 'everyone', 'all'])
    elif value == 'e 10+ (for everyone 10 and older)':
        rating_poss_vals.extend(['e 10+', 'e 10 plus', 'everyone 10', 'everyone above', 'everyone over', 'everyone older'])
    elif value == 't (for teen)':
        rating_poss_vals.extend(['t rated', 'rated t', 't rating', 'rating t', 'teen', 'teens', 'teenagers'])
    elif value == 'm (for mature)':
        rating_poss_vals.extend(['m rated', 'rated m', 'm rating', 'rating m', 'mature', 'adult'])

    sent_tok = word_tokenize(sent.lower())

    for rating_val in rating_poss_vals:
        if rating_val.count(' ') == 0:
            idx = find_first_in_list(rating_val, sent_tok)
            if idx >= 0:
                # if len(rating_val) == 1:
                #     return sent.find(rating_val)
                # else:
                return sent.find(rating_val)
        else:
            pos = sent.find(rating_val)
            if pos >= 0:
                return pos

    return pos


def ratingSlot(sent, value):
    pos = -1
    rating_poss_vals = [value]

    if value == 'excellent':
        rating_poss_vals.extend(['amazing', 'awesome', 'fantastic', 'great', 'high'])
    elif value == 'good':
        rating_poss_vals.extend(['acclaim', 'fun', 'positive', 'solid', 'well'])
    elif value == 'average':
        rating_poss_vals.extend(['decent', 'mediocre', 'middle', 'middling', 'moderate', 'okay'])
    elif value == 'poor':
        rating_poss_vals.extend(['bad', 'disappointing', 'lackluster', 'low', 'negative', 'poorly'])

    for rating_val in rating_poss_vals:
        pos = sent.find(rating_val)
        if pos >= 0:
            return pos

    return pos


def hasMultiplayerSlot(sent, value):
    sent = re.sub('-', ' ', sent.lower())
    sent_tok = word_tokenize(sent)

    for sval in ['multiplayer', 'friends']:
        idx = find_first_in_list(sval, sent_tok)
        if idx >= 0:
            pos = sent.find(sval)
            if value == 'no':
                for negation in negation_cues:
                    if negation in sent_tok:
                        neg_idxs = find_all_in_list(negation, sent_tok)
                        for neg_idx in neg_idxs:
                            if idx - neg_idx < NEG_THRESH:
                                return pos
            else:
                return pos

    # Alternative for when the value is 'no'
    if value == 'no':
        pos = sent.find('single player')
        if pos >= 0:
            return pos

    return -1


def availableOnSteamSlot(sent, value):
    sent_tok = word_tokenize(sent.lower())

    idx = find_first_in_list('steam', sent_tok)
    if idx >= 0:
        pos = sent.find('steam')
        if value == 'no':
            for negation in negation_cues:
                if negation in sent_tok:
                    neg_idxs = find_all_in_list(negation, sent_tok)
                    for neg_idx in neg_idxs:
                        if idx - neg_idx < NEG_THRESH:
                            return pos
        else:
            return pos

    return -1


def hasLinuxReleaseSlot(sent, value):
    sent_tok = word_tokenize(sent.lower())

    idx = find_first_in_list('linux', sent_tok)
    if idx >= 0:
        pos = sent.find('linux')
        if value == 'no':
            for negation in negation_cues:
                if negation in sent_tok:
                    neg_idxs = find_all_in_list(negation, sent_tok)
                    for neg_idx in neg_idxs:
                        if idx - neg_idx < NEG_THRESH:
                            return pos
            for negation_phr in negation_phrases:
                neg_pos = sent.find(negation_phr)
                if neg_pos >= 0:
                    neg_idxs = find_all_in_list(negation_phr.split(' ')[0], sent_tok)
                    for neg_idx in neg_idxs:
                        if idx - neg_idx < NEG_THRESH:
                            return pos
        else:
            return pos

    return -1


def hasMacReleaseSlot(sent, value):
    sent_tok = word_tokenize(sent.lower())

    idx = find_first_in_list('mac', sent_tok)
    if idx >= 0:
        pos = sent.find('mac')
        if value == 'no':
            for negation in negation_cues:
                if negation in sent_tok:
                    neg_idxs = find_all_in_list(negation, sent_tok)
                    for neg_idx in neg_idxs:
                        if idx - neg_idx < NEG_THRESH:
                            return pos
            for negation_phr in negation_phrases:
                neg_pos = sent.find(negation_phr)
                if neg_pos >= 0:
                    neg_idxs = find_all_in_list(negation_phr.split(' ')[0], sent_tok)
                    for neg_idx in neg_idxs:
                        if idx - neg_idx < NEG_THRESH:
                            return pos
        else:
            return pos

    return -1


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
            synset_ctr = 0

            for synset in synsets:
                synset_ctr += 1
                hypernyms = synset.hypernyms()

                # If none of the first 3 meanings of the word has "food" as hypernym, then we do not want to
                #   identify the word as food-related (e.g. "center" has its 14th meaning associated with "food",
                #   or "green" has its 7th meaning accociated with "food").
                while synset_ctr <= 3 and len(hypernyms) > 0:
                    lemmas = [l.name() for l in hypernyms[0].lemmas()]

                    if 'food' in lemmas:
                        # DEBUG PRINT
                        # print(token)

                        return sent.find(token)
                    # Skip false positives (e.g. "a" in the meaning of "vitamin A" has "food" as a hypernym,
                    #   or "coffee" in "coffee shop" has "food" as a hypernym). There are still false positives
                    #   triggered by proper nouns containing a food term, such as "Burger King" or "The Golden Curry".
                    elif 'vitamin' in lemmas:
                        break
                    elif 'beverage' in lemmas:
                        break

                    # Follow the hypernyms recursively up to the root
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

    slot_root = reduce_slot_name(slot)
    slot_root_plural = get_plural(slot_root)

    if slot_root in curr_tokens or slot_root_plural in curr_tokens or slot in curr_tokens:
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
        
    if reduce_slot_name(slot) in curr_tokens:
        for x in ["information", "info", "inform", "results", "requirement", "requirements", "specification", "specifications"]:
            if x in curr_tokens and ("no" in curr_tokens or "not" in curr_tokens):
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
                        slot_pos = eatTypeSlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True
                    elif slot_root == 'food':
                        slot_pos = foodSlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True
                    elif slot_root == 'pricerange':
                        slot_pos = priceRangeSlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True
                    elif slot_root == 'customerrating':
                        slot_pos = customerRatingSlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True
                    elif slot_root == 'area':
                        slot_pos = areaSlot(sent, value)
                        if slot_pos >= 0:
                            found_slot = True
                    elif slot_root == 'familyfriendly':
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

                    elif slot_root == 'playerperspective':
                        if playerPerspectiveSlot(sent, value) >= 0:
                            found_slot = True
                    elif slot_root == 'genres':
                        if genreSlot(sent, value) >= 0:
                            found_slot = True
                    elif slot_root == 'platforms':
                        if platformsSlot(sent, value) >= 0:
                            found_slot = True
                    elif slot_root == 'esrb':
                        if esrbSlot(sent, value) >= 0:
                            found_slot = True
                    elif slot_root == 'rating':
                        if ratingSlot(sent, value) >= 0:
                            found_slot = True
                    elif slot_root == 'hasmultiplayer':
                        if hasMultiplayerSlot(sent, value) >= 0:
                            found_slot = True
                    elif slot_root == 'availableonsteam':
                        if availableOnSteamSlot(sent, value) >= 0:
                            found_slot = True
                    elif slot_root == 'haslinuxrelease':
                        if hasLinuxReleaseSlot(sent, value) >= 0:
                            found_slot = True
                    elif slot_root == 'hasmacrelease':
                        if hasMacReleaseSlot(sent, value) >= 0:
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
                elif value == "dontcare":
                    if dontcareRealization(sent, slot_root, value):
                        slot_cnt = sent.count(reduce_slot_name(slot_root))
                        if slot_cnt > 1:
                            num_slot_overgens += slot_cnt - 1
                        found_slot = True
                elif value == "none":
                    if noneRealization(sent, slot_root, value):
                        slot_cnt = sent.count(reduce_slot_name(slot_root))
                        if slot_cnt > 1:
                            num_slot_overgens += slot_cnt - 1
                        found_slot = True

                elif slot_root == "name":
                    for pronoun in ["it", "its", "it's", "they"]:
                        if pronoun in word_tokenize(curr_utterance.lower()):
                            found_slot = True
                elif slot_root == "pricerange":
                    # if scorePriceRangeNaive(sent, value):
                    if priceRangeSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "familyfriendly":
                    if familyFriendlySlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "food":
                    if foodSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "area":
                    if areaSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "eattype":
                    if eatTypeSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "customerrating":
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

                elif slot_root == "playerperspective":
                    if playerPerspectiveSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "genres":
                    if genreSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "platforms":
                    if platformsSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "esrb":
                    if esrbSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "rating":
                    if ratingSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "hasmultiplayer":
                    if hasMultiplayerSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "availableonsteam":
                    if availableOnSteamSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "haslinuxrelease":
                    if hasLinuxReleaseSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "hasmacrelease":
                    if hasMacReleaseSlot(sent, value) >= 0:
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
                    if priceRangeSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == 'familyfriendly':
                    if familyFriendlySlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == 'food':
                    if foodSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == 'area':
                    if areaSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == 'eattype':
                    if eatTypeSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == 'customerrating':
                    if scoreCustomerRatingNaive(sent, value) >= 0:
                        found_slot = True

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

                elif slot_root == "playerperspective":
                    if playerPerspectiveSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "genres":
                    if genreSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "platforms":
                    if platformsSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "esrb":
                    if esrbSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "rating":
                    if ratingSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "hasmultiplayer":
                    if hasMultiplayerSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "availableonsteam":
                    if availableOnSteamSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "haslinuxrelease":
                    if hasLinuxReleaseSlot(sent, value) >= 0:
                        found_slot = True
                elif slot_root == "hasmacrelease":
                    if hasMacReleaseSlot(sent, value) >= 0:
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
            slot_pos = eatTypeSlot(utt, value)
        elif slot_root == 'food':
            slot_pos = foodSlot(utt, value)
        elif slot_root == 'pricerange':
            slot_pos = priceRangeSlot(utt, value)
        elif slot_root == 'customerrating':
            slot_pos = customerRatingSlot(utt, value)
        elif slot_root == 'area':
            slot_pos = areaSlot(utt, value)
        elif slot_root == 'familyfriendly':
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

        elif slot_root == "playerperspective":
            slot_pos = playerPerspectiveSlot(utt, value)
        elif slot_root == "genres":
            slot_pos = genreSlot(utt, value)
        elif slot_root == "platforms":
            slot_pos = platformsSlot(utt, value)
        elif slot_root == "esrb":
            slot_pos = esrbSlot(utt, value)
        elif slot_root == "rating":
            slot_pos = ratingSlot(utt, value)
        elif slot_root == "hasmultiplayer":
            slot_pos = hasMultiplayerSlot(utt, value)
        elif slot_root == "availableonsteam":
            slot_pos = availableOnSteamSlot(utt, value)
        elif slot_root == "haslinuxrelease":
            slot_pos = hasLinuxReleaseSlot(utt, value)
        elif slot_root == "hasmacrelease":
            slot_pos = hasMacReleaseSlot(utt, value)

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
