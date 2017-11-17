import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
import grammar_check
import pandas as pd
import csv
import string
from random import shuffle
import numpy as np

def get_grammar_check_count(review):
        tool = grammar_check.LanguageTool('en-GB')
        matches = tool.check(review)
        return 100*len(matches)/len(review.split())

def create_unigram(review):
	"""
	Input: Single review string
	Output: List of Unigram tuples
	"""
	token = nltk.word_tokenize(review)
	unigrams = ngrams(token,1)
	return list(unigrams)


def create_bigram(review):
	"""
	Input: Single review string
	Output: List of Bigram tuples
	"""
	token = nltk.word_tokenize(review)
	bigrams = ngrams(token,2)
	return list(bigrams)

def create_trigram(review):
	"""
	Input: Single review strin
	Output: List of Trigram tuples
	"""
	token = nltk.word_tokenize(review)
	trigrams = ngrams(token,3)
	return list(trigrams)

def get_string_stem(input_string): #if you pass a sentence , it will return the stemmed sentence. Also note that it removes capitalization.
	"""
	Input: String word/sentence
	Output: unicode string which is stem of the word
	"""
	ps = PorterStemmer()
	return (ps.stem(unicode(input_string,'utf-8')))

def remove_stopwords(review):
	"""
	Input: String word/sentence
	Output: String with stop words removed

	"""
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(review)
	newTokens=[]
	for x in tokens:
		if x.lower() not in stopwords.words('english'):
			newTokens.append(x)
	result = [" ".join([w.lower() for w in x.split()]).encode('utf-8').strip() for x in newTokens if x.lower() not in stopwords.words('english')]
	return ( " ".join(result))
	
def get_punctuation_count(input_string):
	"""
	Input: String word/sentence
	Output: Count all punctuations
	"""
	count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
	return count(input_string, string.punctuation)

def get_character_count(input_string):
	"""
	Input: String word/sentence
	Output: Count all punctuations
	"""
	count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
	return count(input_string, string.ascii_letters)

def get_exclamation_count(input_string):
	return input_string.count('!')

def get_question_count(input_string):
	return input_string.count('?')

"""
	Create csv for full data
"""
def create_full_csv():
    with open("neg_fake.csv") as f:
        neg_fake = [review for review in f.read()[:-1].split("\n")]
    with open("pos_fake.csv") as f:
        pos_fake = [review for review in f.read()[:-1].split("\n")]
    with open("neg_true.csv") as f:
        neg_true = [review for review in f.read()[:-1].split("\n")]
    with open("pos_true.csv") as f:
        pos_true = [review for review in f.read()[:-1].split("\n")]
        
    data = pos_fake + neg_fake + pos_true + neg_true
    labels = [0] * (len(pos_fake) + len(neg_fake)) + [1] * (len(pos_true) + len(neg_true))
    pos_neg = [1] * len(pos_fake) + [0] * len(neg_fake) + [1] * len(pos_true) + [0] * len(neg_true)
    processed_data = zip(data, pos_neg, labels)

    data_file = open("data_full.csv", "w")
    wr_file = csv.writer(data_file, delimiter = ',')
    for row in processed_data:
        # print(row)
        wr_file.writerow(row)
    data_file.close()
    return processed_data

"""
	Get full data from csv
"""
def get_full_csv(fileName):
	with open(fileName, 'rb') as file:
		full_data = csv.reader(file, delimiter=',')
		L = []
		for row in full_data:
			L.append((row[0], int(row[1]), int(row[2])))
	shuffle(L)
	return L

def split_file():
	with open('data_full.csv', 'rb') as file:
		full_data = csv.reader(file, delimiter=',')
		L = []
		for row in full_data:
			L.append((row[0], int(row[1]), int(row[2])))
	shuffle(L)
	train = L[int(len(L) * .00) : int(len(L) * .90)]
	test = L[int(len(L) * .91) : int(len(L) * 1.00)]
	data_file_train = open("training.csv", "w")
	wr_file = csv.writer(data_file_train, delimiter = ',')
	for row in train:
		wr_file.writerow(row)
	data_file_train.close()
	data_file_test = open("testing.csv", "w")
	wr_file = csv.writer(data_file_test, delimiter = ',')
	for row in test:
		wr_file.writerow(row)
	data_file_test.close()
"""
	Get labels from the data frame
"""
def get_labels(data_frame):
	labels = []
	for row in data_frame:
		labels.append(row[2])

	return labels

"""
	Subset data and labels based on indexes
"""
def subset_data_labels(data, labels, indexes):
	subset_data = []
	subset_labels = []

	for ind in indexes:
		subset_data.append(data[ind])
		subset_labels.append(labels[ind])

	return subset_data, subset_labels

def get_data_and_labels(fileName):
	labels = []
	data = []
	with open(fileName, 'rb') as file:
		full_data = csv.reader(file, delimiter=',')
		for row in full_data:
			data.append((row[0], int(row[1])))
			labels.append(int(row[2]))
	return data, labels

 