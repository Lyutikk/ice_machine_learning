# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 18:42:54 2022

@author: F1
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



datasets = {
            'nsidc_v4'  : 'b', 
            'jaxa'   : 'r',
            'osisaf' : 'g',
            }

idf = pd.read_csv('../input_data/X_small_iifp_jon.csv',
#idf = pd.read_csv('../input_data/X_invalid_jon.csv',
                  sep=';', index_col=[0]
                  )
idf['state'] = 1

invalid = pd.read_csv('../input_data/X_invalid_jon.csv',
                  sep=';', index_col=[0]
                  )
invalid['state'] = 0

idf = pd.concat((idf, invalid), axis=0)

points = pd.read_csv('meta_309_points.csv', sep=';', index_col=[0])

points['p'] = points.index.values
idf['p'] = idf.index // 41

result = idf.join(points, on='p', how='left',
                  rsuffix="_r",
                  )
result['year'] = 1979 + (result.index % 41)
del result['p_r']

fig = plt.figure()
plt.scatter(points.lon, points.lat, s=2, c='k')
for sic_name in ['osisaf', 'nsidc_v4', 'jaxa'][1:2]:
    sic = result.query('dataset == @sic_name')

    plt.scatter(sic.lon, sic.lat, c=datasets[sic_name],
                s=4,
                label=sic_name)
plt.legend()
plt.grid(ls=':')
plt.show()

for asic in datasets.keys(): #asic = "nsidc_v4"
    years = result.query('dataset == @asic').groupby(by=['year']).count()['x0']
    years.plot(label=asic, legend=True, grid=True)