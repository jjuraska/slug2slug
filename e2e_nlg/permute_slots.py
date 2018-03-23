import pandas as pd
from random import shuffle


# set this value to the number of permutations to produce for each MR
num_permutations = 5


# filter function (it basically gives the count of all the slots from a given MR)
def slot_count(s):
    return len(s.split(','))


def permute(df, print_diagnostics=True):
    num_rows = len(df.index)
    print('Number of samples:', num_rows)

    new_df = pd.DataFrame(columns=['mr', 'ref'])

    for _, row in df.iterrows():
        # print the progress of the data expansion
        if print_diagnostics:
            if num_rows % 500 == 0:
                print('Number of samples remaining:', num_rows)
            num_rows -= 1

        slots = row['mr'].split(',')
        # store the original MR
        new_df.loc[len(new_df)] = row

        for i in range(0, num_permutations):
            shuffle(slots)
            new_df.loc[len(new_df)] = [','.join(slots), row['ref']]

    new_df.to_csv('data/rest_e2e/trainset_augm_%d.csv' % num_permutations, index=False)


def main():
    train_file = 'data/rest_e2e/trainset_stylistic_contrast+agreement+apposition+gerund+fronting+subord.csv'
    # train_file = 'data/rest_e2e/trainset_stylistic_thresh_2.csv'

    df = pd.read_csv(train_file)
    df['mr'] = df['mr'].astype('str')
    df['ref'] = df['ref'].astype('str')

    permute(df, print_diagnostics=True)


if __name__ == '__main__':
    main()
