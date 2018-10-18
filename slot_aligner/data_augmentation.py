import os
import io
import string
import json
import pandas as pd
from collections import OrderedDict

import config
import data_loader
from slot_aligner.slot_alignment import split_content, find_alignment, get_scalar_slots


EMPH_TOKEN = config.EMPH_TOKEN
CONTRAST_TOKEN = config.CONTRAST_TOKEN
CONCESSION_TOKEN = config.CONCESSION_TOKEN


def augment_by_utterance_splitting(dataset, filename):
    """Performs utterance splitting and augments the dataset with new pseudo-samples whose utterances are
    one sentence long. The MR of each pseudo-sample contains only slots mentioned in the corresponding sentence.
    Assumes a CSV or JSON file as input.
    """

    if not filename.lower().endswith(('.csv', '.json')):
        raise ValueError('Unexpected file type. Please provide a CSV or JSON file as input.')

    mrs_dicts = []
    data_new = []

    print('Performing utterance splitting on ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, _, _, value_orig = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value_orig

        mrs_dicts.append(mr_dict)

    new_mrs, new_utterances = split_content(mrs_dicts, utterances, filename, permute=False)

    if filename.lower().endswith('.csv'):
        for row, mr in enumerate(new_mrs):
            if len(mr) == 0:
                continue

            mr_str = ', '.join(['{0}[{1}]'.format(slot, value) for slot, value in mr.items()])

            data_new.append([mr_str, new_utterances[row]])

        # Write the augmented dataset to a new file
        filename_out = ''.join(filename.split('.')[:-1]) + '_utt_split.csv'
        pd.DataFrame(data_new).to_csv(os.path.join(config.DATA_DIR, dataset, filename_out),
                                      header=['mr', 'ref'],
                                      index=False,
                                      encoding='utf8')
    elif filename.lower().endswith('.json'):
        for row, mr in enumerate(new_mrs):
            if len(mr) == 0:
                continue

            mr_str = mr.pop('da')
            mr_str += '(' + slot_sep.join(
                ['{0}{1}{2}'.format(key.rstrip(string.digits), val_sep, value) for key, value in mr.items()]
            ) + ')'

            data_new.append([mr_str, new_utterances[row]])

        # Write the augmented dataset to a new file
        filename_out = ''.join(filename.split('.')[:-1]) + '_utt_split.json'
        with io.open(os.path.join(config.DATA_DIR, dataset, filename_out), 'w', encoding='utf8') as f_data_new:
            json.dump(data_new, f_data_new, indent=4)


def augment_with_emphasis(dataset, filename):
    """Augments the MRs with auxiliary tokens indicating that the following slot should be emphasised in the
    generated utterance, i.e. should be mentioned before the name.
    """

    alignments = []

    print('Augmenting MRs with emphasis in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, slot_orig, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value
            mrs[i] = mrs[i].replace(slot_orig, slot)

        alignments.append(find_alignment(utterances[i], mr_dict))

    for i in range(len(utterances)):
        for pos, slot, _ in alignments[i]:
            if slot == 'name':
                break
            mrs[i] = mrs[i].replace(slot, EMPH_TOKEN + '[], ' + slot)

    new_df = pd.DataFrame(columns=['mr', 'ref'])
    new_df['mr'] = mrs
    new_df['ref'] = utterances

    filename_out = ''.join(filename.split('.')[:-1]) + '_augm_emph.csv'
    new_df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


def augment_with_contrast(dataset, filename):
    """Augments the MRs with auxiliary tokens indicating a pair of slots that should be contrasted in the
    corresponding generated utterance.
    """

    contrast_connectors = ['but', 'however', 'yet']
    scalar_slots = get_scalar_slots()

    alignments = []

    print('Augmenting MRs with contrast in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, slot_orig, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value
            mrs[i] = mrs[i].replace(slot_orig, slot)

        alignments.append(find_alignment(utterances[i], mr_dict))

    for i in range(len(utterances)):
        for contrast_conn in contrast_connectors:
            contrast_pos = utterances[i].find(contrast_conn)
            if contrast_pos >= 0:
                slot_before = None
                value_before = None
                slot_after = None
                value_after = None

                for pos, slot, value in alignments[i]:
                    if pos > contrast_pos:
                        if not slot_before:
                            break
                        if slot in scalar_slots:
                            slot_after = slot
                            value_after = value
                            break
                    else:
                        if slot in scalar_slots:
                            slot_before = slot
                            value_before = value

                if slot_before and slot_after:
                    if slot_before in scalar_slots and slot_after in scalar_slots:
                        if scalar_slots[slot_before][value_before] - scalar_slots[slot_after][value_after] == 0:
                            mrs[i] += ', ' + CONCESSION_TOKEN + '[{0} {1}]'.format(slot_before, slot_after)
                        else:
                            mrs[i] += ', ' + CONTRAST_TOKEN + '[{0} {1}]'.format(slot_before, slot_after)

                break

    new_df = pd.DataFrame(columns=['mr', 'ref'])
    new_df['mr'] = mrs
    new_df['ref'] = utterances

    filename_out = ''.join(filename.split('.')[:-1]) + '_augm_contrast.csv'
    new_df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


def augment_with_contrast_tgen(dataset, filename):
    """Augments the MRs with auxiliary tokens indicating a pair of slots that should be contrasted in the
    corresponding generated utterance. The output is in the format accepted by TGen.
    """

    contrast_connectors = ['but', 'however', 'yet']
    scalar_slots = get_scalar_slots()

    alignments = []
    contrasts = []

    print('Augmenting MRs with contrast in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    slot_sep, val_sep, val_sep_closing = data_cont['separators']

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, slot_orig, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_closing)
            mr_dict[slot] = value
            mrs[i] = mrs[i].replace(slot_orig, slot)

        alignments.append(find_alignment(utterances[i], mr_dict))

    for i in range(len(utterances)):
        contrasts.append(['none', 'none', 'none', 'none'])
        for contrast_conn in contrast_connectors:
            contrast_pos = utterances[i].find(contrast_conn)
            if contrast_pos >= 0:
                slot_before = None
                value_before = None
                slot_after = None
                value_after = None

                for pos, slot, value in alignments[i]:
                    if pos > contrast_pos:
                        if not slot_before:
                            break
                        if slot in scalar_slots:
                            slot_after = slot
                            value_after = value
                            break
                    else:
                        if slot in scalar_slots:
                            slot_before = slot
                            value_before = value

                if slot_before and slot_after:
                    if scalar_slots[slot_before][value_before] - scalar_slots[slot_after][value_after] == 0:
                        contrasts[i][2] = slot_before
                        contrasts[i][3] = slot_after
                    else:
                        contrasts[i][0] = slot_before
                        contrasts[i][1] = slot_after

                break

    new_df = pd.DataFrame(columns=['mr', 'ref', 'contrast1', 'contrast2', 'concession1', 'concession2'])
    new_df['mr'] = mrs
    new_df['ref'] = utterances
    new_df['contrast1'] = [tup[0] for tup in contrasts]
    new_df['contrast2'] = [tup[1] for tup in contrasts]
    new_df['concession1'] = [tup[2] for tup in contrasts]
    new_df['concession2'] = [tup[3] for tup in contrasts]

    filename_out = ''.join(filename.split('.')[:-1]) + '_augm_contrast_tgen.csv'
    new_df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


if __name__ == '__main__':
    augment_by_utterance_splitting('rest_e2e', 'trainset_e2e.csv')
    # augment_by_utterance_splitting('tv', 'train.json')

    # augment_with_emphasis('rest_e2e', 'trainset_e2e.csv')

    # augment_with_contrast('rest_e2e', 'trainset_e2e.csv')
    # augment_with_contrast_tgen('rest_e2e', 'trainset_e2e.csv')
