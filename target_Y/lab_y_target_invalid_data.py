# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:57:06 2022

@author: F1
"""

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
