import pickle
import numpy as np
import glob
import copy


def combine_keep_best(beams_folder):
	list_beams = glob.glob(beams_folder+'/*.pkl')
	n_models = len(list_beams)

	beam1 = pickle.load(open(list_beams[0], 'rb'))
	beam2 = pickle.load(open(list_beams[1], 'rb'))

	if len(beam1) != len(beam2):
		print("Beams of unequal size. \n Aborting.")
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
	pickle.dump( beam_new, open( "predictions/beams_dump_combined.pkl", "wb" ) )

def merge_beams(beams_folder):
	list_beams = glob.glob(beams_folder+'/*.pkl')
	n_models = len(list_beams)
	print("Merging "+ str(n_models)+" beams...")

	all_beams = tuple( pickle.load(open(list_beams[i[0]], 'rb')) for i in enumerate(list_beams))

	beam_new = np.concatenate( all_beams, axis = 0) 

	print(list_beams[0]+" -->size " +str(all_beams[0].shape))
	print(list_beams[1]+" -->size " +str(all_beams[1].shape))
	print(list_beams[2]+" -->size " +str(all_beams[2].shape))
	print(list_beams[3]+" -->size " +str(all_beams[3].shape))
	print(list_beams[4]+" -->size " +str(all_beams[4].shape))
	

	if len(all_beams[0]) != len(all_beams[1]):
		print("Beams of unequal size. \n Aborting.")
		exit(-1)

	beam_new = copy.deepcopy(all_beams[0])

	for b2, i in zip(all_beams[1], range(all_beams[1].shape[0])):
		#k is any beam, now take the ith "row"
		i_row = tuple(k[i] for k in all_beams )
		beam_new[i] = np.concatenate( i_row, axis = 0) 

		
	
	print(all_beams[0].shape)
	print(len(all_beams[0][3]))

	print("------")
	print(all_beams[1].shape)
	print(len(all_beams[1][3]))

	print("-------")
	print(all_beams[2].shape)
	print(len(all_beams[2][3]))

	print("-------")
	print(beam_new.shape)
	print(len(beam_new[3]))
	

	
	np.savez('predictions/beams_combined', beam_new)
	pickle.dump( beam_new, open( "predictions/beams_dump_combined.pkl", "wb" ) )


def main():
	beams_folder = 'beams/'
	#combine_keep_best(beams_folder)
	merge_beams(beams_folder)

if __name__ == '__main__':
	main()