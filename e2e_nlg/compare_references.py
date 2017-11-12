import pandas as pd
from math import ceil

def print_group(g1, g2):
	s= ''
	for sen1, sen2 in zip(g1['ref'], g2['ref']):
		s = s + sen1 + " | " + sen2 +'\n'
	print(s+'\n')



def compare(df1, df2):
	diff_counter = 0 

	for (name1, group1), (name2, group2) in zip(df1.groupby('mr'), df2.groupby('mr')):
		for sen1, sen2 in zip(group1['ref'], group2['ref']):

			# if there is at least one difference print the whole group and break
			if sen1 != sen2:
				print_group(group1, group2)
				diff_counter += 1
				break

	#print stats
	print("# of total differecnes = ", str(diff_counter)+ " (" + str( ceil(diff_counter/len(df1) * 100)) + '%)' )



def main():
	####### SET ##########
	new_file = 'data/train_2best_penalizeAnd.csv'
	old_file = 'data/train_2best.csv'
	######################

	csv_path1 =  new_file
	csv_path2 =  old_file

	df1 = pd.read_csv(csv_path1)
	df2  = pd.read_csv(csv_path2)

	df1['mr']=df1['mr'].astype('str')
	df1['ref']=df1['ref'].astype('str')

	df2['mr']=df2['mr'].astype('str')
	df2['ref']=df2['ref'].astype('str')

	if (len(df1) != len(df2)):
		print("Cannot compare files of different sizes.")
	else:
		compare(df1, df2)


if __name__ == '__main__':
	main()