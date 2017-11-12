import pandas as pd
from math import ceil
import subprocess

def print_group(g1, g2):
	s= ''
	for sen1, sen2 in zip(g1['ref'], g2['ref']):
		s = s + sen1 + " | " + sen2 +'\n'
	print(s+'\n')



def compare(df1, df2):
	''' All the references in a group if they differ in at least one selection '''
	diff_counter = 0 

	for (name1, group1), (name2, group2) in zip(df1.groupby('mr'), df2.groupby('mr')):
		for sen1, sen2 in zip(group1['ref'], group2['ref']):

			# if there is at least one difference print the whole group and break
			if sen1 != sen2:
				print_group(group1, group2)
				diff_counter += 1
				break

	#print stats
	print("# of total (group) differecnes = ", str(diff_counter)+ " (" + str( ceil(diff_counter/len(df1) * 100)) + '%)' )

def print_stats(file1, file2):
	p1_and = subprocess.Popen(["grep", "-c", "and", file1], stdout=subprocess.PIPE)
	p2_and = subprocess.Popen(["grep", "-c", "and", file2], stdout=subprocess.PIPE)

	p1_but = subprocess.Popen(["grep", "-c", "but", file1], stdout=subprocess.PIPE)
	p2_but = subprocess.Popen(["grep", "-c", "but", file2], stdout=subprocess.PIPE)


	out1_and, err = p1_and.communicate()
	out2_and, err = p2_and.communicate()

	out1_but, err = p1_but.communicate()
	out2_but, err = p2_but.communicate()


	print("Number of 'and' keyword: %s - %s = %d" %(file1, file2, int(out1_and)-int(out2_and)) )
	print("Number of 'but' keyword: %s - %s = %d" %(file1, file2, int(out1_but)-int(out2_but)) )

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
		print_stats(new_file, old_file)


if __name__ == '__main__':
	main()