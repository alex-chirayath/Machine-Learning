from data_preprocess import *
from nltk.stem.porter import *
    

def create_complete_bigram_list(dataset):
	"""
	Input:  dataset  
	Output: list of all bigrams
	"""
	final_bigram_list=[]
	for row in dataset:
		review=row[0]
		review=remove_stopwords(review)
		review=get_string_stem(review)
		final_bigram_list+=create_bigram(review)
		final_bigram_list=list(set(final_bigram_list))
	return final_bigram_list

def create_complete_unigram_list(dataset):
	"""
	Input:  dataset
	Output: list of all unigrams
	"""
	final_unigram_list=[]
	
	for row in dataset:
		review=row[0]
		review=remove_stopwords(review)
		review=get_string_stem(review)
		final_unigram_list+=create_unigram(review)
		final_unigram_list=list(set(final_unigram_list))
	return final_unigram_list

def create_complete_trigram_list(dataset):
	"""
	Input:  dataset 
	Output: list of all trigrams
	"""
    
	final_trigram_list=[]
	for row in dataset:
		review=row[0]
		review=remove_stopwords(review)
		review=get_string_stem(review)
		final_trigram_list+=create_trigram(review)
		final_trigram_list=list(set(final_trigram_list))
	return final_trigram_list


def get_top_ngram(ngrams_list,top_max_count):
    """
	Input:  ngram list and top counts 
	Output: list of top ngrams
	"""
    most_common_ngram_list=[]
    fdist = nltk.FreqDist(ngrams_list)
    for ngram_count in fdist.most_common(top_max_count):
        most_common_ngram_list.append(ngram_count[0])
    return most_common_ngram_list


def create_vector(row,final_ngram_list, char_cf=False, punc_cf=False, ex_cf=False, ques_cf=False, grammar_cf=False, pos_neg_cf=False):
	"""
	Input:row(review,pos/neg label,output label) and ngram list
	Output: review vector 
	"""
	review=row[0]
	review=remove_stopwords(review)
	review=get_string_stem(review)

	review_unigram=create_unigram(review)
	review_bigram=create_bigram(review)
	review_trigram=create_trigram(review)

	review_ngram=review_unigram+review_bigram+review_trigram

	review_vector=[0]*len(final_ngram_list)

	#Creates a 1/0 representation
	for ngram in review_ngram:
		if ngram in final_ngram_list:
			review_vector[final_ngram_list.index(ngram)]=1

	#Vector representation : [ngram_counts] + char_count+ punc_count +exclamation_count + question_count+grammar_check_count +[label from csv about pos/neg]
	#review_vector+=[get_character_count(row[0])]+[get_punctuation_count(row[0])]+[get_exclamation_count(row[0])]+[get_question_count(row[0])]+[get_grammar_check_count(row[0])]+[row[1]]

	if char_cf == True:
		review_vector += [get_character_count(row[0])]
	if punc_cf == True:
		review_vector += [get_punctuation_count(row[0])]
	if ex_cf == True:
		review_vector += [get_exclamation_count(row[0])]
	if ques_cf == True:
		review_vector += [get_question_count(row[0])]
	if grammar_cf == True:
		review_vector += [get_grammar_check_count(row[0])]
	if pos_neg_cf == True:
		review_vector += [row[1]]

	return review_vector

def print_list(my_list):
	for row in my_list:
		print row

def get_selected_ngrams(uni_count,bi_count,tri_count,dataset):
	"""
	Input:unigram,bigram,trigram count and data frame
	Output: entire vector list
	"""
	final_unigram_list= create_complete_unigram_list(dataset)
	final_bigram_list= create_complete_bigram_list(dataset)
	final_trigram_list= create_complete_trigram_list(dataset)

	final_ngram_list=get_top_ngram(final_unigram_list,uni_count)+get_top_ngram(final_bigram_list,bi_count)+get_top_ngram(final_trigram_list,tri_count)

	return final_ngram_list

def create_vector_list(final_ngram_list, dataset, char_cf=False, punc_cf=False, ex_cf=False, ques_cf=False, grammar_cf=False, pos_neg_cf=False):
	"""
	Input: final_ngram_list, data frame and other flags
	Output: entire vector list
	"""
	vector_list=[]

	for row in dataset:
		#vector_list.append(create_vector(row,final_ngram_list))
		vector_list.append(create_vector(row,final_ngram_list, char_cf, punc_cf, ex_cf, ques_cf, grammar_cf, pos_neg_cf))
	
	return vector_list


"""
print "getting data"
formatted_dataset = get_full_csv("training.csv")
print "got data"
final_ngram_list = get_selected_ngrams(4,4,4,formatted_dataset)
vector_list=create_vector_list(final_ngram_list,formatted_dataset, True, True, True, True, False, True)

print_list(vector_list)
"""
