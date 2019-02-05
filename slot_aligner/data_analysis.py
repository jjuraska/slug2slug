import os
import pandas as pd
from collections import OrderedDict

import config
import data_loader
from slot_aligner.slot_alignment import find_alignment, count_errors, score_alignment


def align_slots(dataset, filename):
    """Aligns slots of the MRs with their mentions in the corresponding utterances."""

    alignments = []
    alignment_strings = []

    print('Aligning slots in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs_orig, utterances_orig = data_cont['data']
    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    # Tokenize utterances
    utterances = [data_loader.preprocess_utterance(utt) for utt in utterances_orig]

    for i, mr in enumerate(mrs_orig):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value

        alignments.append(find_alignment(utterances[i], mr_dict))

    for i in range(len(utterances)):
        alignment_strings.append(' '.join(['({0}: {1})'.format(pos, slot) for pos, slot, _ in alignments[i]]))

    new_df = pd.DataFrame(columns=['mr', 'ref', 'alignment'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances
    new_df['alignment'] = alignment_strings

    filename_out = ''.join(filename.split('.')[:-1]) + '_aligned.csv'
    new_df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


def score_slot_realizations(dataset, filename):
    """Analyzes unrealized and hallucinated slot mentions in the utterances."""

    errors = []
    incorrect_slots = []
    # slot_cnt = 0

    print('Analyzing missing slot realizations and hallucinations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.EVAL_DIR, dataset, filename))
    dataset_name = data_cont['dataset_name']
    mrs_orig, utterances_orig = data_cont['data']
    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    # Tokenize utterances
    utterances = [data_loader.preprocess_utterance(utt) for utt in utterances_orig]

    for i, mr in enumerate(mrs_orig):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value

            # Count auxiliary slots
            # if not re.match(r'<!.*?>', slot):
            #     slot_cnt += 1

        # TODO: get rid of this hack
        # Move the food-slot to the end of the dict (because of delexing)
        if 'food' in mr_dict:
            food_val = mr_dict['food']
            del(mr_dict['food'])
            mr_dict['food'] = food_val

        # Delexicalize the MR and the utterance
        # utterances[i] = data_loader.delex_sample(mr_dict, utterances[i], dataset=dataset_name)

        # Count the missing and hallucinated slots in the utterance
        cur_errors, cur_incorrect_slots = count_errors(utterances[i], mr_dict)
        errors.append(cur_errors)
        incorrect_slots.append(', '.join(cur_incorrect_slots))

    # DEBUG PRINT
    # print(slot_cnt)

    new_df = pd.DataFrame(columns=['mr', 'ref', 'errors', 'incorrect slots'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['errors'] = errors
    new_df['incorrect slots'] = incorrect_slots

    filename_out = os.path.splitext(filename)[0] + ' (errors).csv'
    new_df.to_csv(os.path.join(config.EVAL_DIR, dataset, filename_out), index=False, encoding='utf8')


def score_emphasis(dataset, filename):
    """Determines how many of the indicated emphasis instances are realized in the utterance.
    """

    emph_missed = []
    emph_total = []

    print('Analyzing emphasis realizations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.EVAL_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    for i, mr in enumerate(mrs):
        expect_emph = False
        emph_slots = set()
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_closing)

            # Extract tokens to be emphasized
            if slot == config.EMPH_TOKEN:
                expect_emph = True
            else:
                mr_dict[slot] = value
                if expect_emph:
                    emph_slots.add(slot)
                    expect_emph = False

        alignment = find_alignment(utterances[i], mr_dict)

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
    new_df['mr'] = mrs
    new_df['ref'] = utterances
    new_df['emphasis (missed)'] = emph_missed
    new_df['emphasis (total)'] = emph_total

    filename_out = ''.join(filename.split('.')[:-1]) + '_eval_emph.csv'
    new_df.to_csv(os.path.join(config.EVAL_DIR, dataset, filename_out), index=False, encoding='utf8')


if __name__ == '__main__':
    # align_slots('rest_e2e', 'devset_e2e.csv')
    align_slots('video_game', 'test.csv')

    # score_slot_realizations(os.path.join('predictions-rest_e2e', 'devset'), 'predictions_devset_TRANS_tmp.csv')
    # score_slot_realizations(os.path.join('predictions-rest_e2e', 'testset'), 'predictions_testset_TRANS_tmp.csv')
    # score_slot_realizations(os.path.join('predictions-video_game', 'testset'), 'predictions TRANS beam 4 (8k).csv')

    # score_emphasis('predictions-rest_e2e_stylistic_selection/devset', 'predictions RNN (4+4) augm emph (reference).csv')
