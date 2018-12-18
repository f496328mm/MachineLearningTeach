
import pandas as pd
import sys
import numpy as np
from __future__ import division, absolute_import, print_function
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
rfc = RandomForestClassifier()
rfc = rfc.fit(x, y)
pred = rfc.predict(x)
result = pd.DataFrame()
result['y'] = y
result['pred'] = pred

score = 0
for i in range(len(result)):
    y2 = result.loc[i,'y']
    pred = result.loc[i,'pred']
    if pred == y2:
        score = score + 1
        
print( score/len(result) )     
#print( accuracy_score(y, pred) * 100 )

print('split data to train and test')
train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size = 0.3, random_state = 0)

print('build model')
rfc = RandomForestClassifier()
np.random.seed(0)
rfc = rfc.fit(train_x, train_y)

print('predict')
train_pred = rfc.predict(train_x)
result = pd.DataFrame()
result['y'] = train_y
result['pred'] = train_pred
result.index = range(len(result))

print('compare score')
score = 0
for i in range(len(result)):
    y2 = result.loc[i,'y']
    pred = result.loc[i,'pred']
    if pred == y2:
        score = score + 1  
print( score/len(result) )



print('predict')
test_pred = rfc.predict(test_x)
result = pd.DataFrame()
result['y'] = test_y
result['pred'] = test_pred
result.index = range(len(result))

print('compare score')
score = 0
for i in range(len(result)):
    y2 = result.loc[i,'y']
    pred = result.loc[i,'pred']
    if pred == y2:
        score = score + 1
        
print( score/len(result) )

