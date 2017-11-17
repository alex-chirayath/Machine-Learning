from data_preprocess import *
from feature_creation import *
from naive_bayes import *
from cross_validation import *
from svm import *
from perceptron import *

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

print "in file"
# data = get_full_csv("data_full.csv")
# labels = get_labels(data)

nbc = get_multinomial_NB_classifier()
#perform_kfold(data, labels, nbc, 5, 2000, 0, 0, random_seed = 0)

svm = get_SVM_classifier(0.1,"linear")
# perform_kfold(data, labels, svm, 5, 2000, 0, 0, random_seed = 0)

perc = get_perceptron_classifier('l1',True)
# perform_kfold(data, labels, perc, 5, 2000, 0, 0, random_seed = 0)

print "model made"
char_cf=False
punc_cf=False
ex_cf=False
ques_cf=False
grammar_cf=False
pos_neg_cf=False

data_train, y_train = get_data_and_labels("training.csv")
data_test, y_test = get_data_and_labels("testing.csv")
print "data got"
final_ngram_list = get_selected_ngrams(2000, 0, 0, data_train)

X_train = create_vector_list(final_ngram_list, data_train, char_cf, punc_cf, ex_cf, ques_cf, grammar_cf, pos_neg_cf)
X_test = create_vector_list(final_ngram_list, data_test, char_cf, punc_cf, ex_cf, ques_cf, grammar_cf, pos_neg_cf)
print "features made"

nbc.fit(X_train, y_train)
y_pred = perc.predict(X_test)
print "nbc accuracy_score and f1_score"
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test, y_pred))


svm.fit(X_train, y_train)
y_pred = perc.predict(X_test)
print "svm accuracy_score and f1_score"
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test, y_pred))


perc.fit(X_train, y_train)
y_pred = perc.predict(X_test)
print "perc accuracy_score and f1_score"
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test, y_pred))