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
import random


def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

print "Preprocessing train data..."

X=[]
f = open( 'train_dataset.txt', 'rU' ) #open the file in read universal mode
for line in f:	
	X.append(line)
	
#print len(X)

"""

y=[]
f2 = open( 'dorothea_train.labels.txt', 'rU' ) #open the file in read universal mode
for line in f2:
	y.append(line)	

#print len(y)


X_1=[]
f3 = open( 'dorothea_valid.data.txt', 'rU' ) #open the file in read universal mode
for line in f3:	
	X.append(line)
	
#print len(X_1)


y_1=[]
f4 = open( 'dorothea_valid.labels.txt', 'rU' ) #open the file in read universal mode
for line in f4:
	y.append(line)

"""

f5 = open("dtrain.txt", "w")
#f6 = open("dtrain.labels.txt", "w")
f7 = open("dvalid.txt", "w")
#f8 = open("dvalid.labels.txt", "w")

count = int(math.floor(0.7*len(X)))
l = range(len(X))
inx = random.sample(l,  count)

print inx, len(inx)

remaining = diff(l, inx)

print remaining, len(remaining)

X_1 = []
X_2 = []
#y_1 = []
#y_2 = []
for i in inx:
	X_1.append(X[i])
	#y_1.append(y[i])

for i in remaining:
	X_2.append(X[i])
	#y_2.append(y[i])

for s in X_1:
        f5.write(s)

#for s in y_1:
#        f6.write(s)

for s in X_2:
        f7.write(s)

#for s in y_2:
#        f8.write(s)
