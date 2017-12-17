import pandas as pd
import numpy as np
import os 
import subprocess 

def say():
	df=pd.read_csv('human_evaluation2.csv')
	mr=df['MR'].astype('str') #store column
	s_id=df['sample #'].astype('str')
	ref0=df['0'].astype('str').apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
	ref1=df['1'].astype('str').apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
	ref2=df['2'].astype('str').apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

	for utt0, utt1, utt2, sample in zip(ref0, ref1, ref2, s_id):
		#what_to_say = 'Sample '+ sample+"...Sentence 0..." + utt0 + '...Sentence 1...' + utt1+ '...Sentence 2...' + utt2
		s1 = subprocess.Popen(['say', 'Sample '+ sample], stderr=subprocess.STDOUT)
		s1.communicate()
		print "Sample: "+ sample

		first = False 
		for i in range(10):
			if first == True:
				r = raw_input("Press Enter to continue or r to repeat...")
				if r != 'r':
					break
			s2 =  subprocess.Popen(['say', "...Sentence 0..." + utt0], stderr=subprocess.STDOUT)
			s2.communicate()
			first = True

			

		first = False 
		for i in range(10):
			if first == True:
				r = raw_input("Press Enter to continue or r to repeat...")
				if r != 'r':
					break
			s3 = subprocess.Popen(['say', "...Sentence 1..." + utt1], stderr=subprocess.STDOUT)
			s3.communicate()
			first = True

		first = False 
		for i in range(10):
			if first == True:
				r = raw_input("Press Enter to continue or r to repeat...")
				if r != 'r':
					break
			s4 = subprocess.Popen(['say', "...Sentence 2..." + utt2], stderr=subprocess.STDOUT)
			s4.communicate()
			first = True

		#say_utt = subprocess.Popen(['say', what_to_say], stderr=subprocess.STDOUT)
		#say_utt.communicate()

		raw_input("Press Enter to continue...")



def main():
	say()
if __name__ == '__main__':
	main()