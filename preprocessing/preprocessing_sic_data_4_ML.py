# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:07:04 2022

@author: F1
"""

#import copy
import glob
import os
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xarray as xr


def interpol(x, obj):
    '''
    '''
    cyear = str(obj.variables['time'].values[0])[:4]
    dr = pd.date_range('{}0101 00:00'.format(cyear), 
                       '{}1231 00:00'.format(cyear), freq='1d')    
    times = obj.variables['time'].values
    
    for i in range(x.shape[0]):
        a = x[i, :]
        ts = pd.Series(times, a)
        ts.reindex(dr)
        return ts   


# X-Y geo coordinates limit domain (Kara Sea)
gx1 = 55
gx2 = 106
gy1 = 68
gy2 = 80

years = range(1979, 2020)

datasets = {
            'nsidc_v4'  : 'cdr_seaice_conc', 
            'jaxa'   : 'sic',
            'osisaf' : 'ice_conc',
            }

meta_points_datasets_pool = []
for dataset in datasets.keys():
    val = datasets[dataset]

    datasetPath = r'e:/SIC_BASE/{0:}/{0:}_sic_1995.nc'.format(dataset)
            
    obj = xr.open_dataset(datasetPath) #, **{'FillValue' : -999})
    dat0 = obj.variables[val].fillna(-999).values[1, ...]
    
    if dataset == 'nsidc_v4':
        obj2 = xr.open_dataset(r'e:/SIC_BASE/nsidc0747.nc')
        lons = obj2.variables['longitude'].values[:]
        lats = obj2.variables['latitude'].values[:]
        obj2.close()
    else:
        lons = obj.variables['lon'].values[:]
        lats = obj.variables['lat'].values[:]    
    

#   If land -> 5 elif ocean > -1
    if dataset != 'nsidc_v4':
        dat = np.where(dat0 < -99, 5, -1)
    else:
        dat = np.where((dat0 >= 2.53) & (dat0 <= 2.54), 5, -1)
    
    
    plt.figure()
    cs = plt.contourf(dat)
    plt.colorbar(cs)
    #for dat in [np.where(dat0 != -4, 1, -1), np.where(dat0 != -4, -1, 1)]:
    
    
    x = []
    y = []
    tad = np.zeros(dat.shape) - 1
    gx = np.zeros(dat.shape)
    gy = np.zeros(dat.shape)

    ij = np.where(dat == 5)
    tad[ij] = 5
#    ij = np.where(dat0 == -5)
#    tad[ij] = -5
    
# SEARCH 
    for i in range(1, dat.shape[0] - 1):
        for j in range(1, dat.shape[1] - 1):
            gx[i, j] = j
            gy[i, j] = i
            if dat[i, j] < 0:
                new = dat[i - 1 : i + 2, j - 1: j + 2]
                if np.any(new > 0):
                    x.append(i)
                    y.append(j)
                    tad[i, j] = 1
    
    df_tad = pd.DataFrame(tad, dtype=int)

    df_tad.to_csv('{}_coast_land_ocean_mask.csv'.format(dataset))
    
    
    jj = np.where(((lons > gx1) & (lons < gx2)) & ((lats > gy1) & (lats < gy2))
                  & (tad < 5)
                  )
        
    
    arr = np.column_stack((lons[jj], lats[jj], tad[jj], gx[jj], gy[jj]))
    df = pd.DataFrame(arr, columns=['lon', 'lat', 'landmask', 'gx', 'gy'])
    

# ------------------------------------------------------------
#   Save points in gx-gx domain
    df.to_excel('kara_{}_gridded_data.xlsx'.format(dataset))
#
# ------------------------------------------------------------
    
    jj = np.where(df.landmask > 0)[0]
    
    new = df.iloc[jj, :]
    

#   Three geo regions were selected fot training coastal samples
    ii1 = np.where((new.lon > 60) & (new.lon < 72.25) & 
              (new.lat > 68) & (new.lat < 74)
              )[0]
    arr1 = new.iloc[ii1, :]
    
    ii2 = np.where((new.lon > 73) & (new.lon < 85.5) & 
              (new.lat > 73) & (new.lat < 77)
              )[0]
    arr2 = new.iloc[ii2, :]
    
    ii3 = np.where((new.lon > 87) & (new.lon < 100) & 
              (new.lat > 74) & (new.lat < 77.5)
              )[0]    
    arr3 = new.iloc[ii3, :]
    
    ii = np.concatenate((ii1, ii2, ii3), axis=0)
    
    mdf = new.iloc[ii]
                    
# ------------------------------------------------------------------------
# Save datasets's info with choosen points (~100)     
    meta100_file = 'kara_100_coastal_points_{}.csv'.format(dataset)
    mdf.to_csv(meta100_file, sep=';')
    meta_points_datasets_pool.append(meta100_file)
    
#
# ------------------------------------------------------------------------   
    
# ========================================================================
#  Part III: analyzes sic time series 

    output_sic_ts_file_name = r'{}_{}_2_df_100p_1979_2019.csv'
                                
    
    nc_path = r'e:/SIC_BASE/{}'.format(dataset)

    flist = glob.glob(nc_path + '/*')
    flist = [f for f in flist if not ('2020' in f[-8:])]

#    assert 1==2

    ii = [int(s) for s in mdf.gx.values]
    jj = [int(s) for s in mdf.gy.values]

    if dataset == 'nsidc_v4':
        
        obj7 = xr.open_dataset(r'e:/SIC_BASE/nsidc0747.nc')
        lon = obj7['longitude']
        lat = obj7['latitude']
        obj7.close()
    else:
        obj7 = xr.open_dataset(flist[10])
        lon = obj7['lon']
        lat = obj7['lat']
        obj7.close()

    if dataset == 'nsidc_v4':
        coef = 100.
    else:
        coef = 1.
        
    lons = lon.values[jj, ii]
    lats = lat.values[jj, ii]


    t1 =  time.time()
    for i, f in enumerate(flist[:]):
        cyear = int(f.split('/')[-1].split('_')[-1][:4])
        print(cyear)
        
        obj = xr.open_dataset(f)
        arr = obj.variables[val]   #.values

        arr = arr.values[:, jj, ii]
        
        print(arr.shape)

        if dataset == 'osisaf':
            dr = pd.date_range('{}0101 12:00'.format(cyear), 
                               '{}1231 12:00'.format(cyear), freq='1d')    
        else:
            dr = pd.date_range('{}0101 00:00'.format(cyear), 
                               '{}1231 00:00'.format(cyear), freq='1d')    
            
        times = obj.variables['time'].values
        
        pool = np.zeros((365, arr.shape[1]))   #copy.deepcopy(arr)
        
        for k in range(arr.shape[1]):
            a = arr[:, k]
            ts = pd.Series(index=times, data=a)
            ts = ts.reindex(dr)
            yarr = ts.interpolate(method='linear', 
                                  axis=0, limit_direction='both')
#                print(yarr.shape)
            cond = (yarr.index.month == 2) & (yarr.index.day == 29)        
            yarr = yarr.loc[~cond]
            
            pool[:, k] = yarr.values[:]
            if np.nan in yarr.values:
                print(k, yarr.dropna())
                
        arr = pool[:]
                   
        if arr.shape[0] == 366:
            print('leap')
            arr = np.delete(arr, [59], axis=0)    
        
        arr = arr.T
        arr = coef * np.reshape(arr, (arr.shape[0], 1, arr.shape[1]))
            
        if i == 0:
            box = arr[:]
        else:
            box = np.concatenate((box, arr), axis=1)
        
    
    t2 =  time.time()
    
    """
    ds = xr.Dataset(
            {"cdr": (("pos", "year", "doy"), box),
    #         "lon" : (("pos"), lons),
    #         "lat" : (("pos"), lats),
             "projection" : obj7['projection'],
    #          : (("lon", "lat"), range(100))
             
             },
             coords={
             "year" : range(box.shape[1]),
    #         "year" : years, #range(box.shape[0]),
             "doy" : range(365),
    #         "pos": (obj7['x'].values[jj, ii], obj7['y'].values[jj, ii]) #.values,
             "lon" : lons,
             "lat" : lats,
             "pos" : range(len(lons))
             
    #         "time" : times
             },
            )
       
    ds.to_netcdf(output_name,
                 encoding={'cdr':{'zlib': True},
                           
                           }
                 )    
    
    """
    
    new = np.reshape(box, (box.shape[0]*box.shape[1], box.shape[2]))
    out = pd.DataFrame(new)
    out.to_csv(output_sic_ts_file_name.format('X', dataset), sep=';')
    
    target = pd.DataFrame(np.where(out > 15, 1, 0))
    target.to_csv(output_sic_ts_file_name.format('Y', dataset), sep=';')
    
    
    t3 =  time.time()
    print('Write fnc file {:.2f} s'.format(t3 - t2))
    print('All time{:.2f} s'.format(t3 - t1))
    

# ========================================================================
#  Part IV: combine datasets 2 one sample


dic_box = {}
for prefix in ['X', 'Y']:
    box = []
    for i, dataset in enumerate(datasets):
        file_path = output_sic_ts_file_name.format(prefix, dataset)

        tmp_df = pd.read_csv(file_path, sep=';',
                             index_col=[0])    #, decimal='.')
        print(dataset, tmp_df.shape)
        print(tmp_df.max().max(), tmp_df.min().min())
        if len(box) > 0:
            box = pd.concat((box, tmp_df), axis=0)
        else:
            box = tmp_df[:]

        os.remove(file_path)
    
    if prefix == 'X':
        box.columns = ['x{}'.format(s) for s in range(365)]
#        box = box - box.mean()
        
    else:
        box.columns = ['y{}'.format(s) for s in range(365)]
    
    
    box.reset_index(drop=True, inplace=True)
    dic_box[prefix[:1]] = box

xpool = []
for i, f in enumerate(meta_points_datasets_pool):
    tmp = pd.read_csv(f, sep=';', 
                      index_col=[0])
    tmp['dataset'] = f[24:].split('.')[0]
    if i == 0:
        xpool = tmp[:]
    else:
        xpool = pd.concat((xpool, tmp), axis=0)

xpool = xpool.reset_index()
xpool.to_csv('meta_{}_points.csv'.format(len(xpool)), sep=';')

del xpool

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


cwd = os.getcwd()
os.chdir('../input_data')
dic_box['X'].to_csv('X_jon.csv', sep=';')
dic_box['Y'].to_csv('Y_jon.csv', sep=';')
invalid.to_csv('X_invalid_jon.csv', sep=';')
invalid0.to_csv('X_small_iifp_jon.csv', sep=';')
os.chdir(cwd)