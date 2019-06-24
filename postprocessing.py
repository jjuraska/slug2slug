import os
import json
from nltk.tokenize import word_tokenize, sent_tokenize
from sacremoses import MosesDetokenizer

import config
from slot_aligner.slot_alignment import extract_delex_placeholders, score_alignment


def finalize_utterances(utterances, mrs):
    utterances_final = []

    for i, utt in enumerate(utterances):
        utt_capitalized = capitalize(utt, mrs[i])
        utt_detokenized = detokenize(utt_capitalized)
        utt_relexed = relex(utt_detokenized, mrs[i])
        # utt_pluralized = join_plural_nouns(utt_relexed)     # disable for E2E, Laptop, Hotel
        utterances_final.append(utt_relexed)

    return utterances_final


def finalize_utterance(utterance, mr_dict):
    return relex(detokenize(capitalize(utterance, mr_dict)), mr_dict)


def capitalize(utt, mr_dict, item_sep=', '):
    # Tokenize the utterance and capitalize I's
    utt_tok = [token.capitalize() if token == 'i' else token for token in utt.split()]

    # Capitalize proper nouns contained in values
    for slot in ['area', 'genres', 'platforms', 'esrb']:
        if slot in mr_dict:
            value = mr_dict[slot]

            if len(value) == 0:
                continue

            # Split the slot value into individual items (to account for list slots)
            items = [item.strip() for item in value.split(item_sep)]

            for item in items:
                if not item[0].isupper():
                    continue

                item = ' '.join(word_tokenize(item))

                if len(item) > 4 or ' ' in item:
                    # Replace long and multi-word values in the string representation
                    utt = utt.replace(item.lower(), item)
                else:
                    # Replace short single-word values in the tokenized representation
                    utt_tok = __replace_lowercase_token(item, utt_tok)

    # Merge the capitalizations in the tokenized and string versions of the utterance
    utt_str_tok = utt.split()
    assert(len(utt_str_tok) == len(utt_tok)), 'Utterances do not have matching lengths.'
    utt_tok_merged = []
    for tok1, tok2 in zip(utt_tok, utt_str_tok):
        if tok1[0].isupper():
            utt_tok_merged.append(tok1)
        else:
            utt_tok_merged.append(tok2)

    utt_tok = utt_tok_merged

    # Capitalize proper nouns in the realizations of boolean slots
    if 'availableonsteam' in mr_dict:
        utt_tok = __replace_lowercase_token('Steam', utt_tok)
    if 'haslinuxrelease' in mr_dict:
        utt_tok = __replace_lowercase_token('Linux', utt_tok)
    if 'hasmacrelease' in mr_dict:
        utt_tok = __replace_lowercase_token('Mac', utt_tok)

    # Return tokenized utterance
    return utt_tok


def detokenize(utt_tokenized):
    # Capitalize I's
    # utterance_tokenized = [token.capitalize() if token == 'i' else token for token in utterance.split()]

    # Detokenize the utterance
    detokenizer = MosesDetokenizer()
    utterance_detokenized = detokenizer.detokenize(utt_tokenized, return_str=True)

    # Fix tokens that do not get detokenized automatically
    utterance_detokenized = utterance_detokenized.replace(' n\'t', 'n\'t').replace('( ', '(')

    # Determine sentence boundaries in the utterance
    sentences = sent_tokenize(utterance_detokenized)
    # Capitalize individual sentences
    sentences = [s[0].upper() + s[1:] for s in sentences]

    # Return utterance as a string
    return ' '.join(sentences)


def relex(utterance, mr_dict):
    # Identify all value placeholders
    matches = extract_delex_placeholders(utterance)

    # Replace the value placeholders with the corresponding values from the MR
    fail_flags = []
    for match in matches:
        slot = match.rstrip('_').split('_')[-1]
        if slot in mr_dict.keys():
            utterance = utterance.replace(match, mr_dict[slot])
        else:
            fail_flags.append(slot)

    if len(fail_flags) > 0:
        print('Warning: when relexicalizing, the following slots could not be handled by the MR: ' + str(fail_flags))
        print(utterance)
        print(mr_dict)

    return utterance


def join_plural_nouns(utterance):
    tokens = utterance.split()

    utterance_new = ''
    cur_pos = 0
    while cur_pos < len(tokens):
        if cur_pos < len(tokens) - 1 and tokens[cur_pos + 1] in ['-s', '-es']:
            token_new = tokens[cur_pos] + tokens[cur_pos + 1].lstrip('-')
            cur_pos += 2
        else:
            token_new = tokens[cur_pos]
            cur_pos += 1
            
        utterance_new += token_new + ' '

    return utterance_new.strip()


def __replace_lowercase_token(tok, utt_tokenized):
    tok_lower = tok.lower()
    for i, w in enumerate(utt_tokenized):
        if w == tok_lower:
            utt_tokenized[i] = tok

    return utt_tokenized


def rerank_beams(beams, keep_n=None, keep_least_errors_only=False):
    """Rerank beams by modifying the log-probability of each candidate utterance based on the slot error score
    indicated by the slot aligner. Keep at most n best candidates.
    """

    beams_reranked = []

    with open(os.path.join(config.DATA_DIR, 'test_source_dict.json'), 'r', encoding='utf8') as f_test_mrs_dict:
        mrs = json.load(f_test_mrs_dict)

    step = max(int(len(mrs) * 0.1), 1)
    checkpoints = range(step - 1, len(mrs), step)

    for index in range(len(mrs)):
        # TODO: load and preprocess the MRs properly, not from the JSON file
        # cur_mr = {slot: ' '.join(word_tokenize(val.lower())) for slot, val in mrs[index].items()}
        cur_mr = mrs[index]
        beam_reranked = []

        for utt, log_prob in beams[index]:
            # Calculate the slot error score and use it to adjust the beam log-probabilities
            score = score_alignment(utt, cur_mr)
            beam_reranked.append((utt, log_prob / score, score))

        if keep_least_errors_only:
            # Filter only those utterances that have the least number of errors identified by the slot aligner
            beam_reranked.sort(key=lambda tup: tup[2], reverse=True)
            beam_reranked = [candidate for candidate in beam_reranked if candidate[2] == beam_reranked[0][2]]

        # Rerank utterances by adjusted beam log-probability
        beam_reranked.sort(key=lambda tup: tup[1], reverse=True)

        # Keep at most n candidates
        if keep_n is not None and len(beam_reranked) > keep_n > 0:
            beam_reranked = beam_reranked[:keep_n]

        # Store the reranked beam
        beams_reranked.append(beam_reranked)

        # Print progress status
        if index in checkpoints:
            progress = (index + 1) // step
            print(str(progress * 10) + '% done')

    return beams_reranked
