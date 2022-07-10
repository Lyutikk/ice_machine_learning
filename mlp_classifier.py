# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:08:27 2022

@author: Admin
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


path = r'D:/ml/nsidc/nsidc_v4_2_df_100p_1979_2020_+.csv'
d = pd.read_csv(path, sep=';', decimal='.')

df = pd.DataFrame(d)
df = df.rename(columns={'Unnamed: 0':'key'})
print('\nDataFrame with all points: df\n', 
      df)


path_y = pd.read_csv('D:/ml/nsidc/Y_target_nsidc_v4_2_df_100p_1979_2020.csv', sep=';', decimal='.')
y_targ = pd.DataFrame(path_y)

y_targ = y_targ.rename(columns={'Unnamed: 0':'key'})
print('\nTarget Y: y_targ\n',
      y_targ)


data = pd.merge(df, y_targ, on='key')
print('\nMerged DataFrames: data\n',
      data)

data = data.dropna()


x_col = [c for c in data.columns if c.endswith('x')]
y_col = [c for c in data.columns if c.endswith('y')]


data_train, data_test = train_test_split(data, 
                                         test_size=0.20)

data_train, data_val = train_test_split(data_train, 
                                        test_size=0.20)


print('\nSize of train data: \n', data_train.shape,
      '\nSize of validation data: \n', data_val.shape,
      '\nSize of test data: \n', data_test.shape)


y_tr = data_train[data_train.isin(data_train)][y_col].values
y_val = data_val[data_val.isin(data_val)][y_col].values
y_test = data_test[data_test.isin(data_test)][y_col].values


x_tr = data_train[data_train.isin(data_train)][x_col].values
x_val = data_val[data_val.isin(data_val)][x_col].values
x_test = data_test[data_test.isin(data_test)][x_col].values

print('\n\nY_train: ', y_tr.shape,
      '\nY_validation: ', y_val.shape,
      '\nY_test: ', y_test.shape)

print('\n\nX_train: ', x_tr.shape,
      '\nX_validation: ', x_val.shape,
      '\nX_test: ', x_test.shape)


y_tr = np.delete(y_tr, [0], 1)
y_val = np.delete(y_val, [0], 1)
y_test = np.delete(y_test, [0], 1)



# [    model     ]

print('\nThe training in the progress.... \n')

model = MLPClassifier(hidden_layer_sizes=[10],
                    random_state=1, 
                    max_iter=300,
                    activation='logistic').fit(x_tr, y_tr[:, :])

print('\n', model, '\nThe training process are completed. \n')


# Predict validation data 


# exp_y = y_val
predict_y_val = model.predict(x_val)


target_names = ['class 0', 'class 1']

print('\nF1 score of validation data: \n', f1_score(y_val, predict_y_val, average='weighted'))
# print(metrics.classification_report(exp_y, predict_y))

# print('\nClassification report(val data): \n', classification_report(y_val, predict_y, target_names=target_names))

print('\nPrecision score(val data): \n', metrics.precision_score(y_val, predict_y_val, average='weighted'))

print('\n Recall score(val data): \n', metrics.recall_score(y_val, predict_y_val, average='weighted'))

print('\nPrecision recall fscore support(val data): \n', metrics.precision_recall_fscore_support(y_val[0], predict_y_val[0], beta=1))
# assert 1==2
print('\nPredicted validation array: \n', pd.DataFrame(predict_y_val))

# fig = plt.figure(figsize=(8, 6))
# plt.plot(predict_y[0, :], alpha=1., color='g', zorder=100)
# plt.grid(ls=':')
# plt.xlabel('DoY')
# plt.ylabel('SiC, %')
# plt.show()


predict_y_test = model.predict(x_test)

print('\nF1 score of test data: \n', f1_score(y_test, predict_y_test, average='weighted'))

# print('\nClassification report(test data): \n', classification_report(y_test, predict_y, target_names=target_names))

print('\nPrecision score(test data): \n', metrics.precision_score(y_test, predict_y_test, average='weighted'))

print('\n Recall score(test data): \n', metrics.recall_score(y_test, predict_y_test, average='weighted'))

print('\nPrecision recall fscore support(test data): \n', metrics.precision_recall_fscore_support(y_test[0], predict_y_test[0], beta=1))

print('\nPredicted test array: \n', pd.DataFrame(predict_y_test))


fig = plt.figure(figsize=(8, 6))
plt.plot(predict_y_test[0, :], alpha=1., color='b', zorder=100)
plt.grid(ls=':')
plt.xlabel('DoY')
plt.ylabel('Y target')
plt.show()








