import io
import re
import numpy as np
import networkx as nx
import pickle
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


# beam retrieval adopted from Shubham Agarwal's code

def get_utterances_from_beam(beam_data):
    vocab_target_file = 'data/vocab_target.txt'
    beam_file = beam_data
    beam_dump_file = 'predictions/beams_dump.pkl'
    
    token_unk = 'UNK'
    token_seq_start = 'SEQUENCE_START'
    token_seq_end = 'SEQUENCE_END'
    
    beam_sequences = []
    beams = np.load(beam_file)

    # load the target vocabulary from file
    vocab_target = []
    with io.open(vocab_target_file, 'r', encoding='utf8') as f_vocab_target:
        vocab_target = f_vocab_target.readlines()

    # extract the first column (containing words), and ignore the second column (containing counts)
    vocab_target = [line.split('\t')[0] for line in vocab_target]

    # add auxiliary tokens to the vocabulary
    vocab_target += [token_unk, token_seq_start, token_seq_end]

    # for predicted_ids, parent_ids, scores in data_iterator:
    for idx in range(len(beams['predicted_ids'])):
    #for idx in range(12):
        prediction_ids = beams['predicted_ids'][idx]
        parent_ids = beams['beam_parent_ids'][idx]
        scores = beams['scores'][idx]
        
        beam_graph = rebuild_graph(prediction_ids, parent_ids, scores, vocab_target)
    
        pred_end_node_names = [pos for pos, node in beam_graph.node.items()
                               if node['name'] == token_seq_end
                                   and len(beam_graph.predecessors(pos)) > 0
                                   and beam_graph.node[beam_graph.predecessors(pos)[0]]['name'] != token_seq_end]
        
        # retrieve the full sequences (omit the start and end tokens)
        sequences = [(tuple(get_path_to_root(beam_graph, pos)[1:-1][::-1]), float(beam_graph.node[pos]['score']))
                     for pos in pred_end_node_names]
    
        # sort the sequences by their score/probability in a decreasing order
        sequences_sorted = sorted(sequences, key=lambda x: x[1], reverse=True)

        probs = np.exp(np.array(list(zip(*sequences_sorted))[1]))
        probs_norm = probs / np.sum(probs)

        sequences_w_prob = [(path, score, prob) for (path, score), prob in zip(sequences_sorted, probs_norm)]
        beam_sequences.append(np.array(sequences_w_prob))
    
    beam_dump = np.array(beam_sequences)
    with open(beam_dump_file, 'wb') as f_beam_dump:
        pickle.dump(beam_dump, f_beam_dump)


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

        # add a new node to the graph
        graph.add_node(new_node)
        graph.node[new_node]['name'] = names[i]
        graph.node[new_node]['score'] = str(scores[i])
        graph.node[new_node]['size'] = 100
        
        # connect the new node with its parent
        graph.add_edge(parent_node, new_node)
