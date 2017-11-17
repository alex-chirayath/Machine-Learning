from sklearn.linear_model import perceptron

def get_perceptron_classifier(penalty_param,fitIntercept):
	return perceptron.Perceptron(penalty=penalty_param,fit_intercept = fitIntercept,)
