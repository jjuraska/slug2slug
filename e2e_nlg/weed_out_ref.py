import nltk
import heapq
from collections import Counter
import pandas as pd
from math import ceil

def eval_ref(group, n, penalize_and):	
	''' Evaluate a group of references and return the n best '''	
	score_tracker = []

	for line in group['ref']:
		pos = nltk.pos_tag(nltk.word_tokenize(line))
		if len(pos) < 2 : #less than two words in mr
			avg_score =  -10000
		else:
			c = Counter([j for i,j in pos])
			word_counter = Counter(line.split(" "))

			#Favor if it has conjuctions and few periods

			# do not penalize if 'and' appears as a conjuction
			if penalize_and == False:
				avg_score = c['CC'] - c['.'] 
			else:
				avg_score = c['CC'] - word_counter['and']*0.5 - c['.']


			#favour if it doesnt start with a proper noun
			if pos[0][1] != 'NNP' and pos[0][1] != 'NNPS':
				if pos[1][1] != 'NNP' and pos[1][1] != 'NNPS':
					#print('# of CC, Periods, + 3 NNP = ', c['CC'], c['.'])
					avg_score =  avg_score  + 2
				else:
					avg_score =  avg_score  + 1


		# invert since this is a min heap
		heapq.heappush(score_tracker, (-avg_score, line))

	#return n smallest since we save the negative of the score since this is a min heap
	return heapq.nsmallest(n, score_tracker)
			
		
def keep_the_best(df,n, penalize_and = False):
	''' Keeps n best references after evaluation '''
	new_df =  pd.DataFrame(columns=['mr', 'ref'])
	for name, group in df.groupby('mr'):
		#print(name)

		n_best = eval_ref(group, n, penalize_and)
		s=''
		for element in n_best:
			#s = s + "S: "+ str(-element[0])+", "+element[1]+'\n'
			new_pair = {'mr': [name], 'ref': [element[1]]}
			temp_df = pd.DataFrame(data=new_pair, dtype=str)
			new_df = new_df.append(temp_df)
		#print(s+'\n')

	new_df.to_csv('train_2best_penalizeAnd.csv', index=False, encoding='utf-8')

def keep_the_best_weighted(df, weight, n):
	''' Increases the weight of the n best references. 
		Note that the weight means that an extra amount of weight many
		instances will be added in addition to the existing ones 
	'''

	new_df =  pd.DataFrame(columns=['mr', 'ref'])
	for name, group in df.groupby('mr'):

		n_best = eval_ref(group, n)
		new_df = new_df.append(group) #append existing

		#TO DO: make this more generic for n best not just 2
		first_best_score = - n_best[0][0]
		second_best_score = -n_best[1][0]

		# if there is a clear winner
		if first_best_score > second_best_score:

			for i in range(weight):
				new_pair = {'mr': [name], 'ref': [n_best[0][1]]}
				temp_df = pd.DataFrame(data=new_pair, dtype=str)
				new_df = new_df.append(temp_df)
		else:
			for i in range(ceil(weight/n)):
				new_pair = {'mr': [name], 'ref': [n_best[0][1]]} #this has to change if n != 2
				temp_df = pd.DataFrame(data=new_pair, dtype=str)
				new_df = new_df.append(temp_df)

			for i in range(ceil(weight/n), weight):
				new_pair = {'mr': [name], 'ref': [n_best[1][1]]}
				temp_df = pd.DataFrame(data=new_pair, dtype=str)
				new_df = new_df.append(temp_df)

	new_df.to_csv('data/train_2best_weight.csv', index=False, encoding='utf-8')





def main():
	#path_ref = 'all_ref.txt'
	#eval_ref(path_ref)

	csv_path =  'data/trainset_e2e.csv'
	df=pd.read_csv(csv_path)
	#df=pd.read_csv('data/trainset.csv', encoding="latin-1")
	df['mr']=df['mr'].astype('str')
	df['ref']=df['ref'].astype('str')

	keep_the_best(df, n=2, penalize_and = True)
	#keep_the_best_weighted(df, weight = 5, n = 2)



	


if __name__ == '__main__':
	main()