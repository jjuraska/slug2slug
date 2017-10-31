import pickle
import numpy as np
import glob


def combine(beams_folder):
	list_beams = glob.glob(beams_folder+'/*.pkl')
	n_models = len(list_beams)

	beam1 = pickle.load(open(list_beams[0], 'rb'))
	

def main():
	beams_folder = 'beams/'
	combine(beams_folder)

if __name__ == '__main__':
	main()