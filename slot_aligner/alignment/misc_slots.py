import re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet


def expReleaseDateSlot(sent, value):
    date = value.replace(';', ',')

    pos = sent.find(date)

    return pos


def developerSlot(sent, value):
    developer = value.replace(';', ',')

    pos = sent.find(developer)

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
