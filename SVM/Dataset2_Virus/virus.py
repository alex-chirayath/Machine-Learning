from sklearn.datasets import load_svmlight_file
import csv
import pandas as pd
from sklearn.decomposition import PCA
import pylab as pl
import numpy as np
import math
from sklearn import svm
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from scipy import sparse
from sklearn import metrics
from ggplot import *



traindata = load_svmlight_file("dtrain.txt", n_features = 531)

testdata = load_svmlight_file("dvalid.txt", n_features = 531)


X= traindata[0]		#train data
y= traindata[1]		#train labels
X_1=testdata[0]			#test data 
y_1=testdata[1]			#test labels 



#---------------------------------------------To scale the training data size--------------------
#uncomment the following lines to scale down the data
xx=X.toarray()


#X=sparse.csr_matrix(xx[0:len(xx)/#<scale factor>])
#y=y[0:len(y)/<scale factor>]

#----------------------------------------------------------------------------------------------------------





# The classifier is running on RBF kernel currently.
# TO run the rbf classifier , comment the Rbf kernel classifier command and uncomment the linear kernal


#-----------------------------------------SVM Classifier------------------------------------------------------------

print "Running SVM Classifier..."
svc = svm.SVC(kernel='linear', C=0.03125,probability=True)

#svc = svm.SVC(C=1, kernel='rbf', degree=2, gamma='auto', coef0=0.0, shrinking=True, probability=True,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
"""
print "Running Feature Selection..."

clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC())),
  ('classification', svc)
])
"""

#To run without feature selection ,uncomment the follwing line
clf=svc


clf.fit(X, y)


#-------ROC CURVE---------------

#clf.probability=True

preds = svc.predict_proba(X_1)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_1, preds)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
graph= ggplot(df, aes(x='fpr', y='tpr')) +  geom_line( color="green" ,size =5) +  geom_abline(linetype='dashed')
print graph


#AUC Curve

auc = metrics.auc(fpr,tpr)

print auc
#----------------------------


#Here we have calcuated score on training data itself as there is  no test data available.

print "Acuuracy of the model -"


print clf.score(X,y)
print clf.score(X_1, y_1)


#-------------------------------Precision and Recall----------------------------
print "Calculating Precision and Recall..."
y_2=clf.predict(X_1)
y_3=np.array(y_1)


false_pos=0
false_neg=0
true_pos=0
true_neg=0

i =0
for eachrow in y_3:
	if eachrow !=y_2[i]:
		if y_2[i] ==1:
			false_pos=false_pos+1
		else:
			false_neg=false_neg+1
	else:
		if y_2[i]==1:
			true_pos=true_pos+1
		else:	
			true_neg=true_neg+1
	i=i+1		



if (true_pos+false_pos)==0:
	print "Precision is not defined in this case"
else :
	precision =(1.0*true_pos)/(true_pos+false_pos)
	print precision


if (true_pos+false_neg)==0:
	print "Recall is not defined in this case"
else :
	recall=(1.0*true_pos)/(true_pos+false_neg)
	print recall


print ""


"""

#To run without cross validation, uncomment the entire section code till the end
# We have initalised the values for C and Gamma 


#------------------------ Cross Validation with Grid Search---------------------

print "Running cross validation with grid Search..."

#	Generates K (training, validation) pairs from the items in X.

#	Each pair is a partition of X, where validation is an iterable
#	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

#	If random is true, a copy of X is shuffled before partitioning,
#	otherwise its order is preserved in training and validation.



def k_fold_crossval(X, Y, K, random = False):
	if random: from random import shuffle; X=list(X); shuffle(X)
	for k in xrange(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		train_labels = [y for i, y in enumerate(Y) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		valid_labels = [y for i, y in enumerate(Y) if i % K == k]
		yield training, train_labels, validation, valid_labels

# ------------------Generating list of C and Gamma-------------------------------------------

c=-5
g=-15
c_list=[]
gamma_list=[]

print "Generating recommended exponential values for C and Gamma for Cross Validation..."
while (c!=17):	
	c_list.append(2**c)			#2^-5 <=C<= 2^15
	c=c+2

while (g!=5):						
	gamma_list.append(2**g)			#2^-15<=Gamma<=2^3
	g=g+2
	
print "C list is"
print c_list
print "Gamma list is"
print gamma_list

print""


results_size = (len(c_list), len(gamma_list))
results = np.zeros(results_size, dtype = np.float)

feature_model =  SelectFromModel(LinearSVC())


print "Running Cross Validation with Grid Search...(***This may take some time***)"

def my_func(c, gamma, training, train_labels, validation, valid_labels):
	
	#svc = svm.SVC(kernel='linear', C=1)
	svc = svm.SVC(C=c, kernel='rbf', degree=2, gamma=gamma, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, 		class_weight=None, verbose=False, max_iter=-1, random_state=None)

	

	print 'c,gamma are----'
	print c,gamma
	#clf=svc
	#print (training)

	svc.fit(training, train_labels)

	#print clf.named_steps['feature_selection'].get_support()

	score = svc.score(validation, valid_labels)	
	print score
	
	return score
	

final = []
size = len(y)
for c_idx in range(len(c_list)):
	for gamma_idx in range(len(gamma_list)):
		total = 0  		
		for training, train_labels, validation, valid_labels in k_fold_crossval(X.toarray(), y, K=3, random = True):
			#tune value of k above for cross validation
    			c = c_list[c_idx]
    			gamma = gamma_list[gamma_idx]
    
    			score = my_func(c, gamma, training, train_labels, validation, valid_labels)		
			total = total + (score*len(valid_labels))	
		score = (total*1.0)/size

		results[c_idx, gamma_idx] = score
	   		


print results
max_index = np.argmax(results)

row = max_index/len(c_list)
col = max_index % len(gamma_list)

print 'Ideal C and gamma are'
print c_list[row], gamma_list[col]

print 'Accuracy with ideal C and gamma'
print np.max(results)
print "Where k is 3	"






"""
