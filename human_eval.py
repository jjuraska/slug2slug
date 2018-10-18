import os
import io
import random
import json
import pandas as pd


def create_eval_files():
    #test_file = 'data/rest_e2e/devset_e2e.csv'
    test_file = 'data/rest_e2e/testset_e2e.csv'

    #num_instances = 547     # for devset_e2e.csv
    num_instances = 630     # for testset_e2e.csv
    num_samples = 40
    
    prediction_files = os.listdir('eval/predictions')
    header = ['sample #', 'MR']
    data = []

    # generate random sample of indexes
    sample_idxs = random.sample(range(0, num_instances), num_samples)
    data.append(sample_idxs)

    # map the filenames onto random numbers
    file_idxs_random = [i for i in range(len(prediction_files))]
    random.shuffle(file_idxs_random)

    files_dict = {}
    for i, filename in enumerate(prediction_files):
        files_dict[file_idxs_random[i]] = filename

    # store the file index map into a file to be used as a key
    with open('eval/file_map.json', 'w') as f_file_map:
        json.dump(files_dict, f_file_map, indent=4, sort_keys=True)

    # sample the MRs
    data_frame_test = pd.read_csv(test_file, header=0, encoding='utf8')
    mrs = data_frame_test.iloc[:, 0].tolist()
    mrs_reduced = []

    for i in range(len(mrs)):
        if i == 0 or mrs[i] != mrs[i - 1]:
            mrs_reduced.append(mrs[i])

    mrs_sampled = [mrs_reduced[idx] for idx in sample_idxs]
    data.append(mrs_sampled)

    # sample the predictions
    for i, filename in enumerate(prediction_files):
        header.append(file_idxs_random[i])
        with io.open(os.path.join('eval', 'predictions', filename), 'r', encoding='utf8') as f_predictions:
            predictions = f_predictions.read().splitlines()
            predictions_sampled = [predictions[idx] for idx in sample_idxs]
            data.append(predictions_sampled)

    # create a data frame
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = header

    # reorder the columns so the file indexes are in an increasing order
    header_reordered = header[:2] + [i for i in range(len(prediction_files))]
    df = df[header_reordered]

    # store the data frame into a CSV file
    df.to_csv('eval/human_evaluation.csv', index=False, encoding='utf8')


if __name__ == "__main__":
    create_eval_files()
