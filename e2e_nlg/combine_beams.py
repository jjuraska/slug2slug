import pickle
import numpy as np
import glob
import copy


def combine_keep_best(beams_folder):
    list_beams = glob.glob(beams_folder + '/*.pkl')
    n_models = len(list_beams)

    beam1 = pickle.load(open(list_beams[0], 'rb'))
    beam2 = pickle.load(open(list_beams[1], 'rb'))

    if len(beam1) != len(beam2):
        print('Error: Beam files of unequal length.\nAborting.')
        exit(-1)

    n_sentences = len(beam1)
    beam_new = copy.deepcopy(beam1) #start as a copy of beam1 (by value not reference)
    # beam structure : n_sentences x sentence per beam
    # each sentence is list  = [token, logits, score]
    for b1, b2, i in zip(beam1, beam2, range(beam1.shape[0])):
        #print(b1[1][1])
        for sentence1, sentence2, k in zip(b1, b2, range(len(b1))):
            if sentence1[1] < sentence2[1]:
                beam_new[i][k]= sentence2

    np.savez('predictions/beams_combined', beam_new)
    pickle.dump( beam_new, open('predictions/beams_dump_combined.pkl', 'wb'))


def merge_beams(beams_folder):
    list_beams = glob.glob(beams_folder + '/*.pkl')

    num_models = len(list_beams)
    if num_models == 0:
        print('Error: No beam files found.')
        exit(-1)

    beams = []
    for beam_file in list_beams:
        beams.append(pickle.load(open(beam_file, 'rb')))

    for i in range(num_models - 1):
        if len(beams[i]) != len(beams[i + 1]):
            print('Error: Beam files of unequal length.\nAborting.')
            exit(-1)

    beams_combined = np.copy(beams[0])
    for i in range(len(beams[0])):
        beams_combined[i] = np.concatenate((beams[0][i], beams[1][i]), axis=0)
        

    #print(beam1.shape)
    #print(len(beam1[0]))
    #print('------')
    #print(beam2.shape)
    #print(len(beam2[0]))
    #print('-------')
    #print(beam3.shape)
    #print(len(beam3[0]))
    #print('-------')
    print(beams_combined.shape)
    print(len(beams_combined[0]))

    
    np.savez('predictions/beams_combined', beams_combined)
    pickle.dump(beams_combined, open('predictions/beams_dump_combined.pkl', 'wb'))


def main():
    beams_folder = 'beams/rest_e2e/'
    #beams_folder = 'beams/tv/'
    #beams_folder = 'beams/laptop/'

    #combine_keep_best(beams_folder)
    merge_beams(beams_folder)

if __name__ == '__main__':
    main()