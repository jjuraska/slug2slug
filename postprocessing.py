import os
import io
import re
import numpy as np
import json
import pickle
import networkx as nx
from nltk.tokenize import word_tokenize, sent_tokenize
from sacremoses import MosesDetokenizer

import config
from slot_aligner.slot_alignment import score_alignment


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
    return relex(detokenize(utterance), mr_dict)


def capitalize(utt, mr_dict, item_sep='; '):
    # Tokenize the utterance and capitalize I's
    utt_tok = [token.capitalize() if token == 'i' else token for token in utt.split()]

    # Capitalize proper nouns contained in values
    for slot in ['area', 'genres', 'platforms', 'esrb']:
        if slot in mr_dict:
            value = mr_dict[slot]

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
    matches = re.findall(r'<slot_.*?>', utterance)

    # Replace the value placeholders with the corresponding values from the MR
    fail_flags = []
    for match in matches:
        slot = match.split('_')
        slot = slot[-1].rstrip('>')
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


def align_beams(beams=None, beams_file=None, data_file=None):
    new_beams = []
    if beams is None:
        if beams_file is None:
            beams_file = 'predictions/beams_dump.pkl'
        with open(beams_file, 'rb') as openfile:
            beams = pickle.load(openfile)
            
    # mrs = extractMRs(data_file)
    with io.open(os.path.join(config.DATA_DIR, 'test_source_dict.json'), 'r', encoding='utf8') as f_test_mrs_dict:
        mrs = json.load(f_test_mrs_dict)

    step = max(int(len(mrs) * 0.1), 1)
    checkpoints = range(step - 1, len(mrs), step)

    for index in range(0, len(mrs)):
        curr_mr = mrs[index]
        scored_beams = []

        for beam in beams[index]:
            # utterance = beam[0]
            utterance, logprob, score = beam
            sent = " ".join(utterance)
            score = score_alignment(sent, curr_mr)
            new_beam = np.asarray((beam[0], beam[1] / score, beam[2] * score))      # beam[1] ~ log-prob (negative), beam[2] ~ prob (positive)
            scored_beams.append((score, new_beam))

        scored_beams.sort(key=lambda tup: tup[1][1], reverse=True)
        final_beams = [beam[1] for beam in scored_beams]
        new_beams.append(final_beams)

        # Print progress status
        if index in checkpoints:
            progress = (index + 1) // step
            print(str(progress * 10) + '% done')

    with open('predictions/beams_dump_reranked.pkl', 'wb') as f_beam_dump:
        pickle.dump(np.array(new_beams), f_beam_dump)

    return new_beams


def align_beams_t2t(beams=None, beams_file=None, data_file=None):
    new_beams = []
    if beams is None:
        if beams_file is None:
            beams_file = 'predictions/beams_dump.pkl'
        with open(beams_file, 'rb') as openfile:
            beams = pickle.load(openfile)

    # mrs = extractMRs(data_file)
    with io.open(os.path.join(config.DATA_DIR, 'test_source_dict.json'), 'r', encoding='utf8') as f_test_mrs_dict:
        mrs = json.load(f_test_mrs_dict)

    step = max(int(len(mrs) * 0.1), 1)
    checkpoints = range(step - 1, len(mrs), step)

    for index in range(len(mrs)):
        cur_mr = mrs[index]
        scored_beams = []

        for utt, log_prob in beams[index]:
            score = score_alignment(utt, cur_mr)
            scored_beams.append((utt, log_prob / score))

        scored_beams.sort(key=lambda x: x[1], reverse=True)
        new_beams.append(scored_beams)

        # Print progress status
        if index in checkpoints:
            progress = (index + 1) // step
            print(str(progress * 10) + '% done')

    # with open('predictions/beams_dump_reranked.pkl', 'wb') as f_beam_dump:
    #     pickle.dump(np.array(new_beams), f_beam_dump)

    return new_beams


# Beam retrieval adopted from Shubham Agarwal's code
def get_utterances_from_beam(beam_data):
    vocab_target_file = 'data/vocab_target.txt'
    beam_file = beam_data
    beam_dump_file = 'predictions/beams_dump.pkl'
    
    token_unk = 'UNK'
    token_seq_start = 'SEQUENCE_START'
    token_seq_end = 'SEQUENCE_END'

    beam_sequences = []
    beams = np.load(beam_file)

    # Load the target vocabulary from file
    vocab_target = []
    with io.open(vocab_target_file, 'r', encoding='utf8') as f_vocab_target:
        vocab_target = f_vocab_target.readlines()

    # Extract the first column (containing words), and ignore the second column (containing counts)
    vocab_target = [line.split('\t')[0] for line in vocab_target]

    # Add auxiliary tokens to the vocabulary
    vocab_target += [token_unk, token_seq_start, token_seq_end]

    step = max(int(len(beams['predicted_ids']) * 0.1), 1)
    checkpoints = range(step - 1, len(beams['predicted_ids']), step)

    # For predicted_ids, parent_ids, scores in data_iterator:
    # for idx in range(12):
    for idx in range(len(beams['predicted_ids'])):
        prediction_ids = beams['predicted_ids'][idx]
        parent_ids = beams['beam_parent_ids'][idx]
        scores = beams['scores'][idx]
        
        beam_graph = rebuild_graph(prediction_ids, parent_ids, scores, vocab_target)
    
        pred_end_node_names = [pos for pos, node in beam_graph.node.items()
                               if node['name'] == token_seq_end
                                   and len(beam_graph.predecessors(pos)) > 0
                                   and beam_graph.node[beam_graph.predecessors(pos)[0]]['name'] != token_seq_end]
        
        # Retrieve the full sequences (omit the start and end tokens)
        sequences = [(tuple(get_path_to_root(beam_graph, pos)[1:-1][::-1]), float(beam_graph.node[pos]['score']))
                     for pos in pred_end_node_names]
    
        # Sort the sequences by their score/probability in a decreasing order
        sequences_sorted = sorted(sequences, key=lambda x: x[1], reverse=True)

        probs = np.exp(np.array(list(zip(*sequences_sorted))[1]))
        probs_norm = probs / np.sum(probs)

        sequences_w_prob = [(path, score, prob) for (path, score), prob in zip(sequences_sorted, probs_norm)]
        beam_sequences.append(np.array(sequences_w_prob))

        # Print progress status
        if idx in checkpoints:
            progress = (idx + 1) // step
            print(str(progress * 10) + '% done')
    
    with open(beam_dump_file, 'wb') as f_beam_dump:
        pickle.dump(np.array(beam_sequences), f_beam_dump)

    return beam_sequences


def get_path_to_root(graph, node_pos):
    predecessor = graph.predecessors(node_pos)
    assert len(predecessor) <= 1

    self_seq = [graph.node[node_pos]['name'].split('\t')[0]]
    if len(predecessor) == 0:
        return self_seq
    else:
        return self_seq + get_path_to_root(graph, predecessor[0])


def rebuild_graph(prediction_ids, parent_ids, scores, vocab=None):
    def get_node_name(pred_id):
        return vocab[pred_id] if vocab else str(pred_id)
    
    graph = nx.DiGraph()
    utterance_len = prediction_ids.shape[0]

    for cur_depth in range(utterance_len):
        names = [get_node_name(pred_id) for pred_id in prediction_ids[cur_depth]]
        __extend_graph(graph, cur_depth + 1, parent_ids[cur_depth], names, scores[cur_depth])

    graph.node[(0, 0)]['name'] = 'START'

    return graph


def __extend_graph(graph, depth, parent_ids, names, scores):
    for i, parent_id in enumerate(parent_ids):
        new_node = (depth, i)
        parent_node = (depth - 1, parent_id)

        # Add a new node to the graph
        graph.add_node(new_node)
        graph.node[new_node]['name'] = names[i]
        graph.node[new_node]['score'] = str(scores[i])
        graph.node[new_node]['size'] = 100
        
        # Connect the new node with its parent
        graph.add_edge(parent_node, new_node)


if __name__ == '__main__':
    align_beams(data_file='testset_e2e.csv')
