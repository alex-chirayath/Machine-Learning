from data_preprocess import *
from feature_creation import *
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
def get_kfold_indices(data, labels, k, shuffle_flag, random_seed):
	kf = KFold(n_splits=k, shuffle = shuffle_flag, random_state = random_seed)

	return kf.split(data, labels)

def perform_kfold(data, labels, model, k, uni_count, bi_count, tri_count, shuffle_flag = True, random_seed = 123,
	char_cf=False, punc_cf=False, ex_cf=False, ques_cf=False, grammar_cf=False, pos_neg_cf=False):

	for train_index, test_index in get_kfold_indices(data, labels, k, shuffle_flag, random_seed):
		print "in kfold"
		data_train, y_train = subset_data_labels(data, labels, train_index)
		data_test, y_test = subset_data_labels(data, labels, test_index)

		final_ngram_list = get_selected_ngrams(uni_count, bi_count, tri_count, data_train)
		X_train = create_vector_list(final_ngram_list, data_train, char_cf, punc_cf, ex_cf, ques_cf, grammar_cf, pos_neg_cf)
		X_test = create_vector_list(final_ngram_list, data_test, char_cf, punc_cf, ex_cf, ques_cf, grammar_cf, pos_neg_cf)

		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		print(accuracy_score(y_test,y_pred))
		print(f1_score(y_test, y_pred))



