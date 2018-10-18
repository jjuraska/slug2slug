import nltk
import heapq
import re
import numpy as np
import pandas as pd
from collections import Counter
from math import ceil
from pycorenlp import StanfordCoreNLP


def add_period(line):
    ''' add a period to the end of the reference'''
    if line[-1] != '.':
        line = line + '.'
    return line


def eval_ref(group, n, penalize_and):    
    ''' Evaluate a group of references and return the n best '''    
    score_tracker = []

    for line in group['ref']:

        # add period
        line = add_period(line)

        pos = nltk.pos_tag(nltk.word_tokenize(line))

        if len(pos) < 2:        # less than two words in reference
            avg_score = -10000
        else:
            c = Counter([j for i,j in pos])
            word_counter = Counter(line.split(' '))

            # Favor if it has conjunctions and few periods

            # do not penalize if 'and' appears as a conjunction
            if penalize_and == False:
                avg_score = c['CC'] - c['.']
            else:
                avg_score = c['CC'] - word_counter['and']*0.5 - c['.']

            # favour if it doesnt start with a proper noun
            if pos[0][1] != 'NNP' and pos[0][1] != 'NNPS':
                if pos[1][1] != 'NNP' and pos[1][1] != 'NNPS':
                    #print('# of CC, Periods, + 3 NNP = ', c['CC'], c['.'])
                    avg_score = avg_score + 2
                else:
                    avg_score = avg_score + 1

        # negate since this is a min heap
        heapq.heappush(score_tracker, (-avg_score, line))

    # return n smallest since we save the negative of the score since this is a min heap
    return heapq.nsmallest(n, score_tracker)


def eval_ref_alt(group, n, penalize_and):    
    ''' Evaluate a group of references and return the n best '''
    score_thresh = 2
    scores = []

    nlp = StanfordCoreNLP('http://localhost:9000')

    for utt in group['ref']:
        # add period
        utt = add_period(utt)

        output = nlp.annotate(utt, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse',
            'outputFormat': 'json'
        })

        # join the parse trees of individual sentences in the utterance
        ptree = '\n'.join([sent['parse'] for sent in output['sentences']])

        num_sents = ptree.count('(ROOT')

        # divide the parse tree into lines
        ptree = ptree.split('\n')

        style_score = 0
        if find_contrast(ptree):
            style_score += 3
        if find_agreement(ptree):
            style_score += 3
        if find_apposition(ptree):
            style_score += 2
        if find_fronted_adjective_phrase(ptree):
            style_score += 2
        if find_fronted_prepositional_phrase(ptree):
            style_score += 2
        if find_fronted_verb_phrase(ptree):
            style_score += 2
        # if find_fronted_imperative_phrase(ptree):
        #     style_score += 2
        # if find_modal_verb(ptree):
        #     style_score += 2
        if find_gerund_verb(ptree):
            style_score += 2
        if find_subordinate_clause_non_wh(ptree):
            style_score += 2
        if find_subordinate_clause_wh(ptree):
            style_score += 1
        # if find_existential_there(ptree):
        #     style_score += 1
        # if find_prepositions(ptree):
        #     style_score += 1

        # normalize the score using the sentence count in the utterance
        # style_score /= num_sents
        # style_score -= 0.5 * (num_sents - 1)

        scores.append((style_score, utt))
    
    top_candidates = [(score, utt) for score, utt in scores if score >= score_thresh]
    # if len(top_candidates) == 0:
    #     top_candidates = [max(scores, key=lambda item:item[0])]
    
    return top_candidates


def find_apposition(ptree):
    comma_tag_opening = '(, ,)'
    comma_tag_closing = '(, ,))'
    comma_tag_indent = -1
    parens_match = 0

    for i, line in enumerate(ptree):
        line_content = line.strip()
        if comma_tag_indent == -1:
            if line_content == comma_tag_opening:
                comma_tag_indent = line.find(comma_tag_opening)
                continue
        else:
            if line_content == comma_tag_closing and\
                    parens_match == 0 and\
                    line.find(comma_tag_closing) == comma_tag_indent:
                return True
            elif line_content == comma_tag_opening:
                # call the function recursively on the remainder of the parse tree
                if find_apposition(ptree[i:]):
                    return True
            else:
                parens_match += line_content.count('(') - line_content.count(')')

    return False


def find_fronted_adjective_phrase(ptree):
    sentence_tag = '(ROOT'
    clause_tags = ['(S', '(SINV']
    adj_phrase_tag = '(ADJP'
    is_fronted = False
    clause_tag_cnt = 0
    expect_adj_phrase = False

    for line in ptree:
        line_content = line.strip()
        if line_content == sentence_tag:
            is_fronted = True
        elif is_fronted and line_content in clause_tags:
            clause_tag_cnt += 1
            expect_adj_phrase = True
        elif expect_adj_phrase and line_content.startswith(adj_phrase_tag):
            return True
        else:
            is_fronted = False
            clause_tag_cnt = 0
            expect_adj_phrase = False

    return False


def find_fronted_prepositional_phrase(ptree):
    sentence_tag = '(ROOT'
    clause_tags = ['(S', '(SINV']
    prep_phrase_tag = '(PP'
    is_fronted = False
    clause_tag_cnt = 0
    expect_prep_phrase = False

    for line in ptree:
        line_content = line.strip()
        if line_content == sentence_tag:
            is_fronted = True
        elif is_fronted and line_content in clause_tags:
            clause_tag_cnt += 1
            expect_prep_phrase = True
        elif expect_prep_phrase and line_content.startswith(prep_phrase_tag):
            return True
        else:
            is_fronted = False
            clause_tag_cnt = 0
            expect_prep_phrase = False

    return False


def find_fronted_verb_phrase(ptree):
    sentence_tag = '(ROOT'
    clause_tags = ['(S', '(SINV']
    verb_phrase_tag = '(VP'
    is_fronted = False
    clause_tag_cnt = 0
    expect_verb_phrase = False

    for line in ptree:
        line_content = line.strip()
        if line_content == sentence_tag:
            is_fronted = True
        elif is_fronted and line_content in clause_tags:
            clause_tag_cnt += 1
            if clause_tag_cnt > 1:
                expect_verb_phrase = True
        elif expect_verb_phrase and line_content.startswith(verb_phrase_tag):
            return True
        else:
            is_fronted = False
            clause_tag_cnt = 0
            expect_verb_phrase = False

    return False


def find_fronted_imperative_phrase(ptree):
    sentence_tag = '(ROOT'
    clause_tags = '(S'
    imperative_phrase_tag = '(VP (VB '
    is_fronted = False
    expect_imperative_phrase = False

    for line in ptree:
        line_content = line.strip()
        if line_content == sentence_tag:
            is_fronted = True
        elif is_fronted and line_content in clause_tags:
            expect_imperative_phrase = True
        elif expect_imperative_phrase and line_content.startswith(imperative_phrase_tag):
            return True
        else:
            is_fronted = False
            expect_imperative_phrase = False

    return False


def find_subordinate_clause_non_wh(ptree):
    subord_clause_tag = '(SBAR (IN'

    for line in ptree:
        line_content = line.strip()
        if line_content.startswith(subord_clause_tag):
            return True

    return False


def find_subordinate_clause_wh(ptree):
    subord_clause_tag = '(SBAR'
    wh_phrase_tag = '(WH'
    expect_wh_phrase = False

    for line in ptree:
        line_content = line.strip()
        if line_content == subord_clause_tag:
            expect_wh_phrase = True
        elif expect_wh_phrase and line_content.startswith(wh_phrase_tag):
            return True
        else:
            expect_wh_phrase = False

    return False


def find_gerund_verb(ptree):
    gerund_verb_tag = '(VBG'

    for line in ptree:
        if gerund_verb_tag in line:
            return True

    return False


def find_modal_verb(ptree):
    modal_verb_tag = '(MD'

    for line in ptree:
        if modal_verb_tag in line:
            return True

    return False


def find_contrast(ptree):
    contrast_tags = ['(CC but)'.lower(),
                     '(IN despite)'.lower(),
                     '(RB however)'.lower(),
                     '(RB nevertheless)'.lower(),
                     '(RB yet)'.lower()]

    for line in ptree:
        line = line.lower()
        for contrast_tag in contrast_tags:
            if contrast_tag in line:
                return True

    return False


def find_agreement(ptree):
    agreement_tags = ['(ADVP (RB too))'.lower(),
                      '(ADVP (RB as) (RB well))'.lower(),
                      '(CONJP (RB as) (RB well) (IN as))'.lower(),
                      '(DT both)'.lower(),
                      '(DT either)'.lower(),
                      '(DT neither)'.lower(),
                      '(DT nor)'.lower(),
                      '(RB also)'.lower()]

    for line in ptree:
        line = line.lower()
        for agreement_tag in agreement_tags:
            if agreement_tag in line:
                return True

    return False


def find_existential_there(ptree):
    ex_there_tag = '(EX there)'.lower()

    for line in ptree:
        line = line.lower()
        if ex_there_tag in line:
            return True

    return False


def find_prepositions(ptree):
    prep_tags = ['(PP (IN with)'.lower()]

    for line in ptree:
        line = line.lower()
        for prep_tag in prep_tags:
            if prep_tag in line:
                return True

    return False
            
        
def keep_the_best(df, n, penalize_and=False):
    ''' Keeps n best references after evaluation '''
    new_df = pd.DataFrame(columns=['mr', 'ref', 'score'])

    for mr, group in df.groupby('mr'):
        print(mr)

        n_best = eval_ref_alt(group, n, penalize_and)
        s = ''
        for element in n_best:
            #s = s + "S: "+ str(-element[0])+", "+element[1]+'\n'
            new_pair = {'mr': [mr], 'ref': [element[1]], 'score': [element[0]]}
            temp_df = pd.DataFrame(data=new_pair, dtype=str)
            new_df = new_df.append(temp_df)
        #print(s+'\n')

    if penalize_and == True:
        file_name = 'data/rest_e2e/train_%sbest_penalizeAnd.csv' % (str(n))
    else:
        file_name = 'data/rest_e2e/train_stylistic.csv'

    new_df.to_csv(file_name, index=False, encoding='utf-8')


def keep_the_best_weighted(df, weight, n, penalize_and=False):
    ''' Increases the weight of the n best references. 
        Note that the weight means that an extra amount of weight many
        instances will be added in addition to the existing ones 
    '''

    new_df = pd.DataFrame(columns=['mr', 'ref'])
    for name, group in df.groupby('mr'):

        n_best = eval_ref(group, n, penalize_and)
        new_df = new_df.append(group)   # append existing

        # TODO: make this more generic for n best not just 2
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
                new_pair = {'mr': [name], 'ref': [n_best[0][1]]}    # this has to change if n != 2
                temp_df = pd.DataFrame(data=new_pair, dtype=str)
                new_df = new_df.append(temp_df)

            for i in range(ceil(weight/n), weight):
                new_pair = {'mr': [name], 'ref': [n_best[1][1]]}
                temp_df = pd.DataFrame(data=new_pair, dtype=str)
                new_df = new_df.append(temp_df)

    new_df.to_csv('data/train_2best_weight.csv', index=False, encoding='utf-8')


def test_parsing(df):
    nlp = StanfordCoreNLP('http://localhost:9000')


    utt = 'There is a pub called Wildwood which serves English food. It has a low customer rating and price range - typically less than £20.'

    output = nlp.annotate(utt, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse',
        'outputFormat': 'json'
    })

    # divide the parse tree into lines
    ptree = '\n'.join([sent['parse'] for sent in output['sentences']])
    print(ptree)

    # if find_apposition(ptree.split('\n')):
    # if find_fronted_adjective_phrase(ptree.split('\n')):
    # if find_fronted_prepositional_phrase(ptree.split('\n')):
    # if find_fronted_verb_phrase(ptree.split('\n')):
    # if find_fronted_imperative_phrase(ptree.split('\n')):
    # if find_subordinate_clause_non_wh(ptree.split('\n')):
    # if find_subordinate_clause_wh(ptree.split('\n')):
    # if find_gerund_verb(ptree.split('\n')):
    if find_modal_verb(ptree.split('\n')):
    # if find_contrast(ptree.split('\n')):
    # if find_agreement(ptree.split('\n')):
    # if find_prepositions(ptree.split('\n')):
    # if find_existential_there(ptree.split('\n')):
        print(utt)


    # for utt in df['ref']:
    #     output = nlp.annotate(utt, properties={
    #         'annotators': 'tokenize,ssplit,pos,depparse,parse',
    #         'outputFormat': 'json'
    #     })
    #
    #     # divide the parse tree into lines
    #     ptree = '\n'.join([sent['parse'] for sent in output['sentences']])
    #
    #     # if find_apposition(ptree.split('\n')):
    #     if find_fronted_adjective_phrase(ptree.split('\n')):
    #     # if find_fronted_prepositional_phrase(ptree.split('\n')):
    #     # if find_fronted_verb_phrase(ptree.split('\n')):
    #     # if find_fronted_imperative_phrase(ptree.split('\n')):
    #     # if find_subordinate_clause_non_wh(ptree.split('\n')):
    #     # if find_subordinate_clause_wh(ptree.split('\n')):
    #     # if find_gerund_verb(ptree.split('\n')):
    #     # if find_modal_verb(ptree.split('\n')):
    #     # if find_contrast(ptree.split('\n')):
    #     # if find_agreement(ptree.split('\n')):
    #     # if find_prepositions(ptree.split('\n')):
    #     # if find_existential_there(ptree.split('\n')):
    #         print(utt)
    #         # print()
    #         # print(ptree)


def main():
    # csv_path = 'data/rest_e2e/trainset_e2e.csv'
    csv_path = 'data/rest_e2e/devset_e2e.csv'
    # csv_path = 'eval/predictions-rest_e2e_stylistic_selection/devset/test.csv'
    df = pd.read_csv(csv_path)

    # activate encoding if running locally:
    # df=pd.read_csv('data/trainset_e2e.csv', encoding='latin-1')

    df['mr'] = df['mr'].astype('str')
    df['ref'] = df['ref'].astype('str')

    # test_parsing(df)

    keep_the_best(df, n=2, penalize_and=False)
    # keep_the_best(df, n=2, penalize_and=True)
    # keep_the_best_weighted(df, weight = 5, n = 2)


if __name__ == '__main__':
    main()
