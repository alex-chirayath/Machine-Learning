import csv
import pandas as pd
import numpy as np
import math
from sklearn import svm
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn import metrics
from ggplot import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pylab as pl



#We first initialize the array
#the number for rows depends on the number of samples
#the number of columns is fixed=100000 features


#---------------------- Creating the array---------------------------
print "Preprocessing train data..."

X=[]
f = open( 'dtrain.data.txt', 'rU' ) #open the file in read universal mode
for line in f:
	a=[0]*100000		
	column=[]
	cells = line.split( " " )
	column=np.asarray(cells)
	for index in range(len(column)-1):
		b= long(column[index])
		b=(b)-1
		#print b
		a[b]=1	
	X.append(a)
	
#print len(X)


y=[]
f2 = open( 'dtrain.labels.txt', 'rU' ) #open the file in read universal mode
for line in f2:
	y.append(int(line))	

#print len(y)

#-------------------- Now for Validation data------------------------------------------------

print "Preprocessing validation data..."

X_1=[]
f3 = open( 'dvalid.data.txt', 'rU' ) #open the file in read universal mode
for line in f3:
	a=[0]*100000		
	column=[]
	cells = line.split( " " )
	column=np.asarray(cells)
	for index in range(len(column)-1):
		b= long(column[index])
		b=b-1
		#print b
		a[b]=1	
	X_1.append(a)
	
#print len(X_1)


y_1=[]
f4 = open( 'dvalid.labels.txt', 'rU' ) #open the file in read universal mode
for line in f4:
	y_1.append(int(line))	

#print len(y_1)



#------------------------------------------------------------------------------------------


#---------------------------------------------To scale the training data size--------------------
#uncomment the following lines to scale down the data



#X=X[0:len(X) /20]#<scale factor>]
#y=y[0:len(y) /20]#<scale factor>]

#----------------------------------------------------------------------------------------------------------


# The classifier is running on RBF kernel currently.
# TO run the linear classifier , comment the Rbf kernel classifier command and uncomment the linear kernel


#-----------------------------------------SVM Classifier------------------------------------------------------------

print "Running SVM Classifier..."
#svc = svm.SVC(kernel='linear', C=0.003125,probability=True)

svc = svm.SVC(C=128, kernel='rbf', degree=2, gamma=0.0078125 , coef0=0.0, shrinking=True, probability=True,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)


print "Running Feature Selection..."

# uncomment out whichever feature selection you want to carry out

#clf = Pipeline([
 # ('feature_selection', SelectFromModel(LinearSVC())),
  #('classification', svc)
#])


clf = Pipeline([
  ('feature_selection', PCA(n_components=10)),
  ('classification', svc)
])


#clf = Pipeline([
 # ('feature_selection',SelectKBest(chi2, k=2)),
 # ('classification', svc)
#])

"""
#----------------Data PLOT using PCA=2-----------------

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)


for i in range(0, X_r.shape[0]):
	if y[i]==-1:
		c1 = pl.scatter(X_r[i,0],X_r[i,1],c='r',marker='+')
	if y[i]==1:
		 c2 = pl.scatter(X_r[i,0],X_r[i,1],c='g',marker='o')
pl.legend([c2, c1], ['Class 2', 'Class 1'])
#pl.figure(figsize=(8, 6), dpi=80)
pl.ylim([-5,22])
pl.xlim([-2.5,18])
pl.show()

"""



#To run without feature selection ,uncomment the follwing line
#clf=svc

clf.fit(X, y)

print "Accuracy of the model -"
print clf.score(X, y)
print clf.score(X_1, y_1)







#---------------------------  ROC CURVE ------------------------------------------------

#clf.probability=True

preds = clf.predict_proba(X_1)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_1, preds)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
graph= ggplot(df, aes(x='fpr', y='tpr')) +  geom_line( color="blue" ,size =3) +  geom_abline(linetype='dashed')
print graph

#-------------AUC Curve

auc = metrics.auc(fpr,tpr)
print auc


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

# ------------------Generating list of C and Gamma-------------------------------

c=-5
g=-15
c_list=[]
gamma_list=[]

print "Generating recommended exponential values for C and Gamma for Cross Validation..."
while (c!=15):	
	c_list.append(2**c)			#2^-5 <=C<= 2^15
	c=c+2

while (g!=3):						
	gamma_list.append(2**g)			#2^-15<=Gamma<=2^3
	g=g+2
	
print "C list is"
print c_list

gamma_list=[2]
print "Gamma list is"
print gamma_list




results_size = (len(c_list), len(gamma_list))
results = np.zeros(results_size, dtype = np.float)

feature_model =  SelectFromModel(LinearSVC())


# The classifier is running on RBF kernel currently.
# To run the linear classifier , comment the Rbf kernel classifier command and uncomment the linear kernal


def my_func(c, gamma, training, train_labels, validation, valid_labels):
	
	svc = svm.SVC(kernel='linear', C=c)	
	#svc = svm.SVC(C=c, kernel='rbf', degree=2, gamma=gamma, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, 		class_weight=None, verbose=False, max_iter=-1, random_state=None)

	clf = Pipeline([
  	('feature_selection', PCA(n_components=30)),
  	('classification', svc)
	])

	#clf = Pipeline([
	 # ('feature_selection',SelectKBest(chi2, k=30)),
  	  #('classification', svc)
	#])

	print 'c,gamma are----'
	print c,gamma
	#clf=svc

	clf.fit(training, train_labels)

	#print clf.named_steps['feature_selection'].get_support()


	score = clf.score(validation, valid_labels)	
	print score
	
	return score
	

final = []
size = len(y)
for c_idx in range(len(c_list)):
	for gamma_idx in range(len(gamma_list)):
		total = 0  		
		for training, train_labels, validation, valid_labels in k_fold_crossval(X, y, K=3):
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



print "Running SVM Classifier..."
#svc = svm.SVC(kernel='linear', C=c_list[row],probability=True)

svc = svm.SVC(C=c_list[row], kernel='rbf', degree=2, gamma=gamma_list[col], coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)


print "Running Feature Selection..."


#clf = Pipeline([
 # ('feature_selection', SelectFromModel(LinearSVC())),
  #('classification', svc)
#])


#clf = Pipeline([
#  ('feature_selection', PCA(n_components=100)),
#  ('classification', svc)
#])


clf = Pipeline([
	  ('feature_selection',SelectKBest(chi2, k=30)),
  	  ('classification', svc)
	])

clf.fit(X, y)

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



