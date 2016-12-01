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
pred_nn_1_retrain = pd.read_csv('../../NN_1/NN_retrain_1.csv', index_col=0)
bound_df_retrain(pred_nn_1_retrain)
pred_nn_2_retrain = pd.read_csv('../../NN_2/NN_retrain_2.csv', index_col=0)
bound_df_retrain(pred_nn_2_retrain)
pred_nn_3_retrain = pd.read_csv('../../NN_3/NN_retrain_3.csv', index_col=0)
bound_df_retrain(pred_nn_3_retrain)
pred_nn_4_retrain = pd.read_csv('../../NN_4/NN_retrain_4.csv', index_col=0)
bound_df_retrain(pred_nn_4_retrain)
pred_nn_5_retrain = pd.read_csv('../../NN_5/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_nn_5_retrain)
pred_nn_6_retrain = pd.read_csv('../../XGB_1/XGB_retrain_1.csv', index_col=0)
bound_df_retrain(pred_nn_6_retrain)

from scipy.optimize import minimize

# ======================== NN optimize ======================== #
def f(coord,args):
    pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,r = args
    return np.mean( np.abs(coord[0]*pred_1 + coord[1]*pred_2 + coord[2]*pred_3
                          +coord[3]*pred_4 + coord[4]*pred_5 + coord[5]*pred_6
                          - r))


initial_guess = np.array([0.5 for x in range(6)])


res = minimize(f,initial_guess,args = [
                                        pred_nn_1_retrain['loss'].values,
                                        pred_nn_2_retrain['loss'].values,
                                        pred_nn_3_retrain['loss'].values,
                                        pred_nn_4_retrain['loss'].values,
                                        pred_nn_5_retrain['loss'].values,
                                        pred_nn_6_retrain['loss'].values,
                                       train['loss'].values]
                              ,method='SLSQP')

print res
