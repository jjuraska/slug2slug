import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.moses import MosesDetokenizer


def detokenize_batch(utterances):
    utterances_detokenized = []

    for utterance in utterances:
        utterances_detokenized.append(detokenize(utterance))

    return utterances_detokenized


def detokenize(utterance):
    # capitalize I's
    utterance_tokenized = [token.capitalize() if token == 'i' else token for token in utterance.split()]

    # detokenize the utterance
    detokenizer = MosesDetokenizer()
    utterance_detokenized = detokenizer.detokenize(utterance_tokenized, return_str=True)

    # fix tokens that do not get detokenized automatically
    utterance_detokenized = utterance_detokenized.replace(' n\'t', 'n\'t')

    # determine sentence boundaries in the utterance
    sentences = sent_tokenize(utterance_detokenized)
    # capitalize individual sentences
    sentences = [s[0].upper() + s[1:] for s in sentences]

    return ' '.join(sentences)


def capitalize_batch(utterances):
    utterances_capitalized = []

    with open('data/vocab_proper_nouns.txt', 'r') as f_vocab:
        proper_nouns = sorted(f_vocab.read().splitlines(), key=len, reverse=True)
        for utterance in utterances:
            utterances_capitalized.append(capitalize(utterance, proper_nouns))

    return utterances_capitalized


def capitalize(utterance, proper_nouns):
    for noun in proper_nouns:
        utterance = utterance.replace(noun.lower(), noun)

    return utterance


def relex_utterance(utterance, mr, replace_name=False):
    # parse the slot-value pairs from the MR
    slots = {}
    for slot_value in mr.split(','):
        sep_idx = slot_value.find('[')
        # parse the slot and the value
        slot = slot_value[:sep_idx].strip()
        value = slot_value[sep_idx + 1:-1].strip()
        slots[slot] = value
    
    # identify all value placeholders
    matches = re.findall(r'&slot_val_.*?&', utterance)
    
    # replace the value placeholders with the corresponding values from the MR
    fail_flags = []
    for match in matches:
        slot = match.split('_')
        slot = slot[-1].rstrip('&')
        if slot in list(slots.keys()):
            if slot == 'name' and replace_name:
                new_val = 'It'
            else:
                new_val = slots[slot]
            utterance = utterance.replace(match, new_val)
        else:
            fail_flags.append(slot)

    # capitalize the first letter of each sentence
    utterance = utterance[0].upper() + utterance[1:]
    sent_end = utterance.find(r'-PERIOD-')
    while sent_end >= 0:
        next_sent_beg = sent_end + 2
        if next_sent_beg < len(utterance):
            utterance = utterance[:next_sent_beg] + utterance[next_sent_beg].upper() + utterance[next_sent_beg + 1:]
        
        sent_end = utterance.find(r'-PERIOD-', next_sent_beg)
    
    # replace the period placeholders
    utterance = utterance.replace(r' -PERIOD-', '.')


    if len(fail_flags) > 0:
        print('When relexing, the following slots could not be handled by the MR: ' + str(fail_flags))
        print(utterance)
        print(mr)

    return utterance
