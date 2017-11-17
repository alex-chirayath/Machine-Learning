from sklearn.naive_bayes import MultinomialNB

def get_multinomial_NB_classifier(alpha_par=1.0, fit_prior_par=True, class_prior_par=None):
	return MultinomialNB(alpha=alpha_par, fit_prior=fit_prior_par, class_prior=class_prior_par)