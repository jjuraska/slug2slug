import os
import subprocess
import scipy.stats as stats

import config


class TTest:
    def __init__(self):
        self.aux_ref_file = os.path.join(config.TTEST_DATA_DIR, 'aux_ref.txt')
        self.aux_sys_A_file = os.path.join(config.TTEST_DATA_DIR, 'aux_sys_A.txt')
        self.aux_sys_B_file = os.path.join(config.TTEST_DATA_DIR, 'aux_sys_B.txt')

        self.bleu_sys_A_file = os.path.join(config.TTEST_SCORES_DIR, 'bleu_scores_sys_A.txt')
        self.bleu_sys_B_file = os.path.join(config.TTEST_SCORES_DIR, 'bleu_scores_sys_B.txt')
        self.nist_sys_A_file = os.path.join(config.TTEST_SCORES_DIR, 'nist_scores_sys_A.txt')
        self.nist_sys_B_file = os.path.join(config.TTEST_SCORES_DIR, 'nist_scores_sys_B.txt')

    def paired_t_test(self, scores_sys_A, scores_sys_B):
        '''Perform the paired t-test to evaluate the significance of the difference of scores in two groups.
        '''

        t, p = stats.ttest_rel(a=scores_sys_A, b=scores_sys_B)

        return t, p

    def score_outputs_bleu_nist_e2e(self, data_ref, data_sys_A, data_sys_B):
        '''Calculate BLEU and NIST scores for individual output utterances. Utilizes the
        '''

        bleu_scores_sys_A = []
        bleu_scores_sys_B = []
        nist_scores_sys_A = []
        nist_scores_sys_B = []

        buffer_ref = []
        idx_sys = 0

        if not os.path.exists(config.TTEST_DATA_DIR):
            os.makedirs(config.TTEST_DATA_DIR)
        if not os.path.exists(config.TTEST_SCORES_DIR):
            os.makedirs(config.TTEST_SCORES_DIR)

        for idx_ref in range(len(data_ref)):
            if len(data_ref[idx_ref]) > 0:
                buffer_ref.append(data_ref[idx_ref])
            else:
                # If the data ends unexpectedly, terminate the loop
                if not data_sys_A[idx_sys] or not data_sys_B[idx_sys]:
                    break

                utt_sys_A = data_sys_A[idx_sys]
                utt_sys_B = data_sys_B[idx_sys]

                self.__create_aux_files(buffer_ref, utt_sys_A, utt_sys_B)

                # Calculate the BLEU score
                command_A = config.METRICS_SCRIPT_PATH + ' ' + self.aux_ref_file + ' ' + self.aux_sys_A_file
                metrics_out_A = subprocess.check_output(command_A, shell=True)
                command_B = config.METRICS_SCRIPT_PATH + ' ' + self.aux_ref_file + ' ' + self.aux_sys_B_file
                metrics_out_B = subprocess.check_output(command_B, shell=True)

                # Extract the BLEU scores from the evaluation script's outputs
                bleu_A = self.__extract_bleu_score(metrics_out_A.decode('utf8'))
                if bleu_A:
                    bleu_scores_sys_A.append(bleu_A)

                bleu_B = self.__extract_bleu_score(metrics_out_B.decode('utf8'))
                if bleu_B:
                    bleu_scores_sys_B.append(bleu_B)

                # Extract the NIST scores from the evaluation script's outputs
                nist_A = self.__extract_nist_score(metrics_out_A.decode('utf8'))
                if nist_A:
                    nist_scores_sys_A.append(nist_A)

                nist_B = self.__extract_nist_score(metrics_out_B.decode('utf8'))
                if nist_B:
                    nist_scores_sys_B.append(nist_B)

                buffer_ref = []
                idx_sys += 1

        # DEBUG PRINT
        # print('BLEU (sys A):', bleu_scores_sys_A)
        # print('NIST (sys A):', nist_scores_sys_A)
        # print('BLEU (sys B):', bleu_scores_sys_B)
        # print('NIST (sys B):', nist_scores_sys_B)

        # Save the BLEU scores to file
        with open(self.bleu_sys_A_file, 'w') as f_bleu_sys_A:
            for bleu in bleu_scores_sys_A:
                f_bleu_sys_A.write(str(bleu) + '\n')

        with open(self.bleu_sys_B_file, 'w') as f_bleu_sys_B:
            for bleu in bleu_scores_sys_B:
                f_bleu_sys_B.write(str(bleu) + '\n')

        # Save the NIST scores to file
        with open(self.nist_sys_A_file, 'w') as f_nist_sys_A:
            for nist in nist_scores_sys_A:
                f_nist_sys_A.write(str(nist) + '\n')

        with open(self.nist_sys_B_file, 'w') as f_nist_sys_B:
            for nist in nist_scores_sys_B:
                f_nist_sys_B.write(str(nist) + '\n')

        # Clean up
        os.remove(self.aux_ref_file)
        os.remove(self.aux_sys_A_file)
        os.remove(self.aux_sys_B_file)

    def __create_aux_files(self, buffer_ref, utt_sys_A, utt_sys_B):
        '''Create auxiliary files of utterances corresponding to a single MR.
        '''

        with open(self.aux_ref_file, 'w') as f_aux_ref:
            with open(self.aux_sys_A_file, 'w') as f_aux_sys_A:
                with open(self.aux_sys_B_file, 'w') as f_aux_sys_B:
                    f_aux_ref.write('\n'.join(buffer_ref))
                    f_aux_sys_A.write(utt_sys_A + '\n')
                    f_aux_sys_B.write(utt_sys_B + '\n')

    def __extract_bleu_score(self, metrics_out):
        '''Parse the BLEU score out of the evaluation script's output.
        '''

        for line in metrics_out.splitlines():
            if line.startswith('BLEU: '):
                line = line.lstrip('BLEU: ')
                try:
                    return float(line)
                except ValueError:
                    return -1.0

        return -1.0

    def __extract_nist_score(self, metrics_out):
        '''Parse the NIST score out of the evaluation script's output.
        '''

        for line in metrics_out.splitlines():
            if line.startswith('NIST: '):
                line = line.lstrip('NIST: ')
                try:
                    return float(line)
                except ValueError:
                    return -1.0

        return -1.0


def main():
    data_ref_file = os.path.join(config.TTEST_DATA_DIR, 'references_rest_e2e_dev.txt')
    # data_sys_A_file = os.path.join(config.TTEST_DATA_DIR, 'predictions_cnn_pool_3_08.5k.txt')
    # data_sys_B_file = os.path.join(config.TTEST_DATA_DIR, 'predictions_cnn_pool_3_utt_split_10k.txt')
    data_sys_A_file = os.path.join(config.TTEST_DATA_DIR, 'predictions_rnn_4+4_09k.txt')
    data_sys_B_file = os.path.join(config.TTEST_DATA_DIR, 'predictions_rnn_4+4_utt_split_16k.txt')

    bleu_sys_A_file = os.path.join(config.TTEST_SCORES_DIR, 'bleu_scores_sys_A.txt')
    bleu_sys_B_file = os.path.join(config.TTEST_SCORES_DIR, 'bleu_scores_sys_B.txt')
    nist_sys_A_file = os.path.join(config.TTEST_SCORES_DIR, 'nist_scores_sys_A.txt')
    nist_sys_B_file = os.path.join(config.TTEST_SCORES_DIR, 'nist_scores_sys_B.txt')

    ttest = TTest()

    # Calculate individual scores for the output utterances
    with open(data_ref_file, 'r') as f_data_ref,\
            open(data_sys_A_file, 'r') as f_data_sys_A,\
            open(data_sys_B_file, 'r') as f_data_sys_B:
        data_ref = f_data_ref.read().splitlines()
        data_ref.append('')
        data_sys_A = f_data_sys_A.read().splitlines()
        data_sys_B = f_data_sys_B.read().splitlines()

        ttest.score_outputs_bleu_nist_e2e(data_ref, data_sys_A, data_sys_B)

    # Perform t-test for the BLEU scores
    with open(bleu_sys_A_file, 'r') as f_bleu_sys_A,\
            open(bleu_sys_B_file, 'r') as f_bleu_sys_B:
        scores_sys_A = [float(x) for x in f_bleu_sys_A.read().splitlines()]
        scores_sys_B = [float(x) for x in f_bleu_sys_B.read().splitlines()]

        t, p = ttest.paired_t_test(scores_sys_A, scores_sys_B)

        print('---- BLEU ----')
        print('t-statistic:', t)
        print('p-value:', p)

    # Perform t-test for the NIST scores
    with open(nist_sys_A_file, 'r') as f_nist_sys_A,\
            open(nist_sys_B_file, 'r') as f_nist_sys_B:
        scores_sys_A = [float(x) for x in f_nist_sys_A.read().splitlines()]
        scores_sys_B = [float(x) for x in f_nist_sys_B.read().splitlines()]

        t, p = ttest.paired_t_test(scores_sys_A, scores_sys_B)

        print('---- NIST ----')
        print('t-statistic:', t)
        print('p-value:', p)


if __name__ == '__main__':
    main()
