# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:21:56 2022

@author: F1
"""

import time
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
#from sklearn.metrics import classification_report


def to_smooth(z, nDays=15):
    '''
    '''

    N2 = (nDays - 1) // 2
        
    # Smoothing two times
    for kk in range(2):
        rw = z.rolling(nDays, center=True).median()
        rw.iloc[0:N2 + 1] = z[0:N2 + 1]
        rw.iloc[-N2:] = z[-N2:]
        
        z = rw[:]
        

    mdf = z[:]
    
    return mdf


datasets = ['nsidc_v4', 'osisaf', 'jaxa']
input_path = r'd:/SCIENCE/2022/Anna_ML/ML_run/input_data'

dic_box = {}
for prefix in ['X', 'Y_target']:
    box = []
    for i, dataset in enumerate(datasets):
        file_path = '{}/{}_{}_2_df_100p_1979_2020.csv'.format(input_path, prefix, dataset)

        tmp_df = pd.read_csv(file_path, sep=';',
                             index_col=[0])    #, decimal='.')
        print(dataset, tmp_df.shape)
        print(tmp_df.max().max(), tmp_df.min().min())
        if len(box) > 0:
            box = pd.concat((box, tmp_df), axis=0)
        else:
            box = tmp_df[:]
    
    if prefix == 'X':
        box.columns = ['x{}'.format(s) for s in range(365)]
#        box = box - box.mean()
        
    else:
        box.columns = ['y{}'.format(s) for s in range(365)]
    
    box.reset_index(drop=True, inplace=True)
    dic_box[prefix[:1]] = box


# Check for invalid treshold values
inv = []
inv0 = []
treshold = 15
df = dic_box['X']
for i, row in df.iterrows():
    arr = row.values
    ii = np.where(arr < treshold)[0]
    if len(ii) >= 1 and len(ii) < 20:
        inv0.append(i)
    elif len(ii) < 1:
        inv.append(i)
        
print(len(inv), 100.*len(inv)/ len(df))

invalid = df.iloc[inv, :]
invalid0 = df.iloc[inv0, :]
# Reserve some invalid points from training sample
#valid_cond = (~df.index.isin(invalid.index[:700]))
#dic_box['X'] = dic_box['X'][valid_cond]
#dic_box['Y'] = dic_box['Y'][valid_cond]


dic_box['X'].to_csv('X_jon.csv', sep=';')
dic_box['Y'].to_csv('Y_jon.csv', sep=';')
invalid.to_csv('X_invalid_jon.csv', sep=';')
invalid0.to_csv('X_small_iifp_jon.csv', sep=';')


X_train0, X_test, Y_train0, Y_test = train_test_split(dic_box['X'], dic_box['Y'],
                                                      test_size=0.20)
X_train, X_val, Y_train, Y_val = train_test_split(X_train0, Y_train0,
                                                  test_size=0.20)

print('\n\nX_train: ', X_train0.shape,
      '\nX_validation: ', X_test.shape,
      '\nX_test: ', X_val.shape)

# ===================================
# Scale the data
scaler = MinMaxScaler()    #StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pickle.dump(scaler, open("scaler.pkl", 'wb'))
"""

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

"""

#assert 1==2
# ==================================
t1 = time.time()
N = 100
model = MLPClassifier(hidden_layer_sizes=[N],
                      random_state=1, 
                      max_iter=300,
                      activation='logistic')

model.fit(X_train_scaled, Y_train)
pickle.dump(model, open('sic_MLP_{}L.sav'.format(N), 'wb'))

"""
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

=====================
"""
t2 = time.time()
print('The training process is completed. Time : {:.2f}'.format(t2 - t1))



#assert 1==2
# ==================================================
X_val_scaled = scaler.transform(X_val)
predict_Y_val = model.predict(X_val_scaled)

X_test_scaled = scaler.transform(X_test)
predict_Y_test = model.predict(X_test_scaled)

#X_test_scaled = scaler.transform(X_test)
predict_Y_train = model.predict(X_train_scaled)


t = 0
sums = 0
for v, pv in zip(Y_val.values, predict_Y_val):
    t += 1
#    if t % 10 == 0:
#        print('F1 Y_val: ', t, f1_score(v, pv))
    sums += f1_score(v, pv)

print(sums / t)
print(f1_score(Y_val, predict_Y_val, average='weighted'))

'''
t = 0
sums = 0
for v, pv in zip(Y_test.values, predict_Y_test):
    t += 1
#    if t % 10 == 0:
#        print('F1 Y_val: ', t, f1_score(v, pv))
    sums += f1_score(v, pv)

print(sums / t)
'''

#print(f1_score(Y_test, predict_Y_test, average='weighted'))
print(f1_score(Y_test, predict_Y_test, average='samples'))   # Equals to (sums / t)

print(f1_score(Y_val, predict_Y_val, average='samples'))   # Equals to (sums / t)
print(f1_score(Y_train, predict_Y_train, average='samples'))   # Equals to (sums / t)


# Fast_check
jj=900
x_test_scaled = scaler.fit_transform(invalid.iloc[jj:jj+1, :])
res = model.predict(x_test_scaled)

fig = plt.figure()
xx = range(365)
plt.plot(xx, invalid.iloc[jj:jj+1, :].values[0, ...], c='b')
plt.plot(xx, res[0, ...]*100., c='r')
plt.grid()

# =======================================================================
# TESTS
# =======================================================================
# I.

#x_test_scaled = scaler.fit_transform(X_val.iloc[jj:jj+1, :])
res = model.predict(X_val_scaled)

xxx = [ix for ix in invalid.index if ix in X_val.index]
xxx_train = [ix for ix in invalid.index if ix in X_train.index]

pic_path = r'd:/SCIENCE/2022/Anna_ML/output/invalid_test'
save_str = ('{0:}'
            '/with_{1:}_invalid_from_{2:}_in_ML_{3:}op'
            '.jpeg'
            )

assert 3==4
print(len(xxx))
if len(xxx) > 1:
    for ii, jj in enumerate(xxx[:]):
#        ii = 2
#        jj = xxx[ii]
        
        fig = plt.figure()
        xx = range(1, 366)
        plt.plot(xx, X_val.loc[jj, :], c='b', label='X_val')
#        plt.plot(xx, 100.*Y_val.loc[jj, :], c='g', label='Y_val')
        plt.plot(xx, 100.*res[ii, :], c='r', label='Y_val_predict')
        plt.grid(ls=':')
        
        plt.xlabel('DoY')
        plt.xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
        plt.xlim(0, 366)

        plt.ylabel('SIC, %')
        plt.yticks(range(0, 101, 10))
        plt.ylim(-5, 105)
        
        plt.legend()
        
        plt.tight_layout()
        

        plt.savefig(save_str.format(pic_path, len(xxx_train), len(invalid), jj),
                    format='jpeg', dpi=400, bbox_inches='tight')
        plt.close()
        
# ============================================================================        
# II. Smooth using N-days median rolling filter
NW = 7
NI = 9
x2 = invalid0.iloc[NI, :].rolling(window=NW).median()
x2[:NW] =  invalid0.iloc[NI, :NW]

x3 = 100. * (x2 - x2.min()) / (x2.max() - x2.min())

pout2 = model.predict(x2.values.reshape((1, 365)))
pout3 = model.predict(x3.values.reshape((1, 365)))

fig = plt.figure()
xx = range(1, 366)

plt.plot(xx, invalid0.iloc[NI, :] , c='gray', alpha=0.8,
         label='X_val_origin')

plt.plot(xx, x3.values, c='g', label='X_val_3_sm_normed')

plt.plot(xx, x2.values, c='b', label='X_val_2_smoothed')
#        plt.plot(xx, 100.*Y_val.loc[jj, :], c='g', label='Y_val')
plt.plot(xx, 100.*pout2[0, :], c='k', label='Y_val_2_predict')
plt.plot(xx, 100.*pout3[0, :], c='red', label='Y_val_3_predict')

plt.grid(ls=':')

plt.xlabel('DoY')
plt.xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
plt.xlim(0, 366)

plt.ylabel('SIC, %')
plt.yticks(range(0, 101, 10))
plt.ylim(-5, 105)

plt.legend()
       
plt.tight_layout()

plt.savefig(r'd:/SCIENCE/2022/Anna_ML/output/case_II_smoothed_{}w_and_normed_{}op.jpeg'.format(NW, invalid0.iloc[NI,:].name),
            format='jpeg', dpi=400, bbox_inches='tight'
            )

