
import pandas as pd
import sys
import numpy as np
from __future__ import division, absolute_import, print_function
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
path = '/home/sam/teach'
sys.path.append(path)

print('load data')
data = pd.read_table('{}/iris.data.txt'.format(path),
                     sep=",",header = None)
data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']

print( 'split x and y, we will predict y by x' )
x = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['class']



print('build model')
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('SVM', SVC()))
scoring = 'accuracy'
# evaluate each model in turn
results = []
names = []
for name, model in models:
	result = model_selection.cross_val_score(
            model, x, y,cv = 3,scoring=scoring)
	results.append(result)
	names.append(name)
	msg = "%s: %f" % (name, result.mean())
	print(msg)

