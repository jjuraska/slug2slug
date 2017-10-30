"""
This file produces the predictions of enseble models by utilizing
inermediate results (i.e. logits) in the logits folder.
"""

import numpy as np
import glob

def add_period(s, token):
	''' Aux function to filter out consecutive periods '''
	if len(s) <= 2 or token != '.':
		return True
	elif token == '.' and s[-1] == '.':
		return False 
	return True


def reshape(m1, m2):
	''' 
	Reshapes matrices in order to concatenate (adds columns of 0s).
	Add 0s not Nan in order to be able to average accros multiple models.
	'''
	print('Reshaping...')
	if m1.shape[1] < m2.shape[1]:
	    #define new shape to store m1 with larger shape[1]
	    new_shape = (m1.shape[0], m2.shape[1], m1.shape[2])
	    resized = np.zeros(new_shape)
	    resized[:m1.shape[0], :m1.shape[1], :m1.shape[2]] = m1
	    return resized,m2

	elif m1.shape[1] > m2.shape[1]:
		new_shape = (m2.shape[0], m1.shape[1], m2.shape[2])
		resized = np.zeros(new_shape)
		resized[:m2.shape[0], :m2.shape[1], :m2.shape[2]] = m2
		return m1,resized

def average(logdir): 
	'''
	This functions looks for all the .npy files in dir and 
	averages the results 
	'''
	list_npy = glob.glob(logdir+'/*.npy')
	n_models = len(list_npy)

	#matrix to store the logits of each model
	cumulative_matrix = None

	for element in list_npy:
		if cumulative_matrix is None:
			print('Loading ' + element)
			cumulative_matrix = np.load(element)
		else:
			print('Loading ' + element)
			temp = np.load(element)
			if temp.shape != cumulative_matrix.shape:
				temp, cumulative_matrix = reshape(temp, cumulative_matrix)
			print('Merging...')
			cumulative_matrix = cumulative_matrix + temp

	#after summing all corresponding values, average them
	print("Averaging..")
	cumulative_matrix = cumulative_matrix / n_models

	print('shape = ', cumulative_matrix.shape)
	return cumulative_matrix

def parse_vocab(vocab_trgt):
	'''
	This function parses the vocab file into a dictionary
	that contains all the mappings of words to numbers 
	'''
	vocab = dict()
	with open(vocab_trgt,'r') as f:
		mapping = 0 #should the map start from 0 or 1???
		for line in f:
			val = str(line).split()
			vocab[mapping] = val[0]
			mapping = mapping + 1

	#append special characters 
	vocab[mapping+1] = 'UNK'
	vocab[mapping+2] = 'SEQUENCE END'
	vocab[mapping+3] = 'SEQUENCE START'

	return vocab

def decode(c_matrix,vocab_trgt):
	'''
	This function decodes the cumulative matrix into text using
	the vocabulary target mappings 
	'''
	n_sentences,_,_ = c_matrix.shape
	mapping_d = parse_vocab(vocab_trgt)
	all_sentences = "" #use this to write to file once 
	for i in range(n_sentences):
		#get sentence 
		s = c_matrix[i, : , :]
		decoded_sentence = ""

		#for each token in the sentence
		for token_arr in s:
			#notice: indexing here starts from 0
			predicted_mapping = np.argmax(token_arr)
			predicted_token = mapping_d[predicted_mapping]
			if predicted_token != 'SEQUENCE END' and add_period(decoded_sentence, predicted_token):
				decoded_sentence = decoded_sentence +" "+predicted_token


		decoded_sentence = decoded_sentence +'\n' #add extra line after sentence is over
		#print("\n Sentence "+str(i)+" :")
		#print(decoded_sentence)

		all_sentences = all_sentences+decoded_sentence

	with open('predictions/predictions.txt','w') as f:
		f.write(all_sentences)

def main():
	c = average('logits_folder')
	decode(c, 'data/vocab_target.txt')
	#d = parse_vocab('data/vocab_target.txt')
	#print d

if __name__ == '__main__':
	main()