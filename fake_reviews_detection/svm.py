from sklearn import svm

def get_SVM_classifier(penalty_param,kernel_type):
	return svm.SVC(C = penalty_param, kernel = kernel_type)
