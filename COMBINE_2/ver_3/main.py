# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np

train = pd.read_csv('../../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
MIN_VALUE = np.min(train['loss'])

def bound_df_retrain(df):
    assert df.shape[0] == 188318
    assert df.columns[0] == 'loss'
    df.loc[df['loss']>MAX_VALUE,:] = MAX_VALUE
    df.loc[df['loss']<MIN_VALUE,:] = MIN_VALUE

### Train ###
pred_retrain_57 = pd.read_csv('../../COMBINE/ver_57/retrain.csv', index_col=0)
bound_df_retrain(pred_retrain_57)
pred_retrain_58 = pd.read_csv('../../COMBINE/ver_58/retrain.csv', index_col=0)
bound_df_retrain(pred_retrain_58)
pred_retrain_60 = pd.read_csv('../../COMBINE/ver_60/retrain.csv', index_col=0)
bound_df_retrain(pred_retrain_60)

pool_list = [pred_retrain_57,
            pred_retrain_58,
            pred_retrain_60
            ]
print "#"
print mean_absolute_error(train['loss'], pred_retrain_57)
print mean_absolute_error(train['loss'], pred_retrain_58)
print mean_absolute_error(train['loss'], pred_retrain_60)
print mean_absolute_error(train['loss'], sum(pool_list)/len(pool_list))

from scipy.optimize import minimize

# ======================== NN optimize ======================== #
def f(coord,args):
    pred_1,pred_2,pred_3,r = args
    return np.mean( np.abs(coord[0]*pred_1 + coord[1]*pred_2 + coord[2]*pred_3
                          - r))


initial_guess = np.array([0.5 for x in range(3)])


res = minimize(f,initial_guess,args = [
                                        pred_retrain_57['loss'].values,
                                        pred_retrain_58['loss'].values,
                                        pred_retrain_60['loss'].values,
                                       train['loss'].values]
                              ,method='SLSQP')

def try_minimize(func, guess, args=(), **kwds):
    '''Minimization of scalar function of one or more variables.
    See the docstring of `scipy.optimize.minimize`.
    '''
    from scipy.optimize import minimize
    from numpy import array2string
    kwds.pop('method', None)

    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']
    results = []
    for method in methods:
        try:
            res = minimize(func, guess, args=args, method=method, options={"maxiter":1000000,"disp":True})
            res.method = method
            results.append(res)
        except ValueError as err:
            print("{:>12s}: {}".format(method, err))
            continue
    print("---------------------------------------------")
    results.sort(key=lambda res:res.fun)
    for res in results:
        out = res.method, str(res.success), res.fun, array2string(res.x, formatter=dict(all=lambda x: "%10.4g" % x), separator=',')
        print("{:>12s}: {:5s}  {:10.5g}  {}".format(*out))
    return results[0]

try_minimize(f, initial_guess, args = [
                                        pred_retrain_57['loss'].values,
                                        pred_retrain_58['loss'].values,
                                        pred_retrain_60['loss'].values,
                                       train['loss'].values])

print res
