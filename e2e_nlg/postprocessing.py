import io
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.moses import MosesDetokenizer


def finalize_utterances(utterances, mrs):
    utterances_final = []

    with io.open('data/vocab_proper_nouns.txt', 'r', encoding='utf8') as f_vocab:
        proper_nouns = sorted(f_vocab.read().splitlines(), key=len, reverse=True)

    for i, utterance in enumerate(utterances):
        utterance_relexed = relex(utterance, mrs[i])
        utterance_capitalized = capitalize(utterance_relexed, proper_nouns)
        utterance_detokenized = detokenize(utterance_capitalized)
        utterances_final.append(utterance_detokenized)

    return utterances_final


def relex(utterance, mr_dict):
    # identify all value placeholders
    matches = re.findall(r'&slot_.*?&', utterance)
    
    # replace the value placeholders with the corresponding values from the MR
    fail_flags = []
    for match in matches:
        slot = match.split('_')
        slot = slot[-1].rstrip('&')
        if slot in mr_dict.keys():
            utterance = utterance.replace(match, mr_dict[slot])
        else:
            fail_flags.append(slot)

    if len(fail_flags) > 0:
        print('Warning: when relexing, the following slots could not be handled by the MR: ' + str(fail_flags))
        print(utterance)
        print(mr_dict)

    return utterance


def capitalize(utterance, proper_nouns):
    for noun in proper_nouns:
        utterance = utterance.replace(noun.lower(), noun)

    return utterance


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
