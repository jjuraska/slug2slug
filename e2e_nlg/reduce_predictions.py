import io
import pandas as pd

predictions_file = 'predictions/predictions.txt'
predictions_reduced_file = 'metrics/predictions_reduced.txt'

with io.open(predictions_file, 'r', encoding='utf8') as f_predictions:
    predictions = f_predictions.read().splitlines()
    
    # create a file with a single prediction for each group of the same MRs
    data_frame_test = pd.read_csv('data/testset.csv', header=0, encoding='utf8')
    test_mrs = data_frame_test.mr.tolist()

    with io.open(predictions_reduced_file, 'w', encoding='utf8') as f_predictions_reduced:
        for i in range(len(test_mrs)):
            if i == 0 or test_mrs[i] != test_mrs[i - 1]:
                f_predictions_reduced.write(predictions[i] + '\n')
