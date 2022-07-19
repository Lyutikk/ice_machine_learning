# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:21:56 2022

@author: F1


https://stackoverflow.com/questions/50565937/how-to-normalize-the-train-and-test-data-using-minmaxscaler-sklearn

Best way is train and save MinMaxScaler model and load the same when it's required.

Saving model:

df = pd.DataFrame({'A':[1,2,3,7,9,15,16,1,5,6,2,4,8,9],'B':[15,12,10,11,8,14,17,20,4,12,4,5,17,19],'C':['Y','Y','Y','Y','N','N','N','Y','N','Y','N','N','Y','Y']})
df[['A','B']] = min_max_scaler.fit_transform(df[['A','B']])  
pickle.dump(min_max_scaler, open("scaler.pkl", 'wb'))
Loading saved model:

scalerObj = pickle.load(open("scaler.pkl", 'rb'))
df_test = pd.DataFrame({'A':[25,67,24,76,23],'B':[2,54,22,75,19]})
df_test[['A','B']] = scalerObj.transform(df_test[['A','B']])


# ================================

https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/


model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

"""

import time
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler #StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


input_path = r'../input_data'
N = 100   # Number of MLP layers
 
# ==========================================================================
# Upload XY-data

dic_box = {}
for prefix in ['X', 'Y']:
    
    box = pd.read_csv('{}/{}_jon.csv'.format(input_path, prefix), sep=';',
                             index_col=[0])    #, decimal='.')
    
    if prefix == 'X':
        box.columns = ['x{}'.format(s) for s in range(365)]
#        box = box - box.mean()
        
    else:
        box.columns = ['y{}'.format(s) for s in range(365)]
    
    box.reset_index(drop=True, inplace=True)
    dic_box[prefix] = box


# ===========================================================================

X_train0, X_test, Y_train0, Y_test = train_test_split(dic_box['X'], 
                                                      dic_box['Y'],
                                                      test_size=0.20)
X_train, X_val, Y_train, Y_val = train_test_split(X_train0,
                                                  Y_train0,
                                                  test_size=0.20)

print('X_train: ', X_train0.shape,
      '\n X_validation: ', X_test.shape,
      '\n X_test: ', X_val.shape)

# ===================================
# Scale the data
scaler = MinMaxScaler()    #StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pickle.dump(scaler, open("scaler.pkl", 'wb'))

# --------------------------------------------------
t1 = time.time()
model = MLPClassifier(hidden_layer_sizes=[N],
                      random_state=1, 
                      max_iter=300,
                      activation='logistic')

model.fit(X_train_scaled, Y_train)

pickle.dump(model, open('sic_MLP_{}L.sav'.format(N), 'wb'))

t2 = time.time()
print('The training process is completed. Time : {:.2f}'.format(t2 - t1))

# --------------------------------------------------

X_val_scaled = scaler.transform(X_val)
predict_Y_val = model.predict(X_val_scaled)

X_test_scaled = scaler.transform(X_test)
predict_Y_test = model.predict(X_test_scaled)

#X_test_scaled = scaler.transform(X_test)
predict_Y_train = model.predict(X_train_scaled)


print('F1 Y_test', f1_score(Y_test, predict_Y_test, average='samples'))   # Equals to (sums / t)
print('F1 Y_val', f1_score(Y_val, predict_Y_val, average='samples'))
print('F1 Y_train', f1_score(Y_train, predict_Y_train, average='samples'))



