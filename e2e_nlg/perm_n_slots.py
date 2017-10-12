import pandas as pd
import os 
import itertools
from random import shuffle

#set this value to the number of permutations to produce for each mr
##############
n_slots=2
##############

#filter function
#it basically gives the count of all the slots from a given MR
def slot_count(s):
	return len(s.split(",")) 


def permute(df, print_diagnostics=True):

	num_rows = len(df.index)
	print ("Number of rows remaining:" + str(num_rows))
	count = 0

	new_df = pd.DataFrame({'mr': [], 'ref': []})
	for d_index, row in df.iterrows():
		r = row['mr'].split(",")
		new_df.loc[len(new_df)] = row #store original

		if print_diagnostics: #prints the progress of the expansion
			if (count == 250):
				print ("Number of rows remaining:" + str(num_rows))
				count = 0 #reset count

			num_rows = num_rows-1
			count += 1

		for j in range(0,n_slots): 
			shuffle(r)
			new_df.loc[len(new_df)] = [  ','.join(r), row['ref'] ]

	new_df.to_csv('trainset_augm_%d.csv' % n_slots, index=False)

def main():
	train_file='data/trainset.csv'
	df=pd.read_csv(train_file)
	#df=pd.read_csv('data/trainset.csv', encoding="latin-1")
	df['mr']=df['mr'].astype('str')
	df['ref']=df['ref'].astype('str')

	permute(df, print_diagnostics=True)



if __name__ == '__main__':
	main()

	