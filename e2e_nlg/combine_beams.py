import pickle
import numpy as np
import glob
import copy


def combine(beams_folder):
	list_beams = glob.glob(beams_folder+'/*.pkl')
	n_models = len(list_beams)

	beam1 = pickle.load(open(list_beams[0], 'rb'))
	beam2 = pickle.load(open(list_beams[1], 'rb'))

	if len(beam1) != len(beam2):
		print("Beams of unequal size. \n Aborting.")
		exit(-1)

	n_sentences = len(beam1)

	print(beam1.shape)
	beam_new = copy.deepcopy(beam1) #start as a copy of beam1 (by value not reference)
	# beam structure : n_sentences x sentence per beam
	# each sentence is list  = [token, logits, score]
	for b1, b2, i in zip(beam1, beam2, range(beam1.shape[0])):
		#print(b1[1][1])
		for sentence1, sentence2, k in zip(b1, b2, range(len(b1))):
			if sentence1[2] < sentence2[2]:
				beam_new[i][k]= sentence2
				


def main():
	beams_folder = 'beams/'
	combine(beams_folder)

if __name__ == '__main__':
	main()