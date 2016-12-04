# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('../../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
# Main
pred_nn_1_retrain = pd.read_csv('../../NN_1/NN_retrain_1.csv', index_col=0)
pred_nn_2_retrain = pd.read_csv('../../NN_2/NN_retrain_2.csv', index_col=0)
pred_nn_3_retrain = pd.read_csv('../../NN_3/NN_retrain_3.csv', index_col=0)
pred_nn_4_retrain = pd.read_csv('../../NN_4/NN_retrain_4.csv', index_col=0)
pred_nn_5_retrain = pd.read_csv('../../NN_5/NN_retrain_5.csv', index_col=0)
pred_nn_6_retrain = pd.read_csv('../../NN_6/NN_retrain_6.csv', index_col=0)
pred_nn_1_fix_retrain = pd.read_csv('../../NN_1_fix/NN_retrain_1.csv', index_col=0)
pred_nn_2_fix_retrain = pd.read_csv('../../NN_2_fix/NN_retrain_2.csv', index_col=0)
pred_nn_3_fix_retrain = pd.read_csv('../../NN_3_fix/NN_retrain_3.csv', index_col=0)
pred_nn_4_fix_retrain = pd.read_csv('../../NN_4_fix/NN_retrain_4.csv', index_col=0)
pred_nn_5_fix_retrain = pd.read_csv('../../NN_5_fix/NN_retrain_5.csv', index_col=0)
pred_nn_6_fix_retrain = pd.read_csv('../../NN_6_fix/NN_retrain_6.csv', index_col=0)
pred_new_nn_1_retrain = pd.read_csv('../../NEW_NN_1/NN_retrain_1.csv', index_col=0)
pred_new_nn_2_retrain = pd.read_csv('../../NEW_NN_2/NN_retrain_2.csv', index_col=0)
pred_new_nn_3_retrain = pd.read_csv('../../NEW_NN_3/NN_retrain_3.csv', index_col=0)
pred_new_nn_3_retrain.loc[pred_new_nn_3_retrain['loss']>MAX_VALUE,:] = MAX_VALUE
pred_new_nn_4_retrain = pd.read_csv('../../NEW_NN_4/NN_retrain_4.csv', index_col=0)
pred_new_nn_4_retrain.loc[pred_new_nn_4_retrain['loss']>MAX_VALUE,:] = MAX_VALUE
pred_new_nn_1_65_retrain = pd.read_csv('../../NEW_NN_1_65/NN_retrain_1.csv', index_col=0)
#sorted(pred_new_nn_3_retrain[130000:150000].values, reverse=True)

pred_nn_1 = pd.read_csv('../../NN_1/NN_1.csv', index_col=0)
pred_nn_2 = pd.read_csv('../../NN_2/NN_2.csv', index_col=0)
pred_nn_3 = pd.read_csv('../../NN_3/NN_3.csv', index_col=0)
pred_nn_4 = pd.read_csv('../../NN_4/NN_4.csv', index_col=0)
pred_nn_5 = pd.read_csv('../../NN_5/NN_5.csv', index_col=0)
pred_nn_6 = pd.read_csv('../../NN_6/NN_6.csv', index_col=0)
pred_nn_1_fix = pd.read_csv('../../NN_1_fix/NN_1.csv', index_col=0)
pred_nn_2_fix = pd.read_csv('../../NN_2_fix/NN_2.csv', index_col=0)
pred_nn_3_fix = pd.read_csv('../../NN_3_fix/NN_3.csv', index_col=0)
pred_nn_4_fix = pd.read_csv('../../NN_4_fix/NN_4.csv', index_col=0)
pred_nn_5_fix = pd.read_csv('../../NN_5_fix/NN_5.csv', index_col=0)
pred_nn_6_fix = pd.read_csv('../../NN_6_fix/NN_6.csv', index_col=0)
pred_new_nn_1 = pd.read_csv('../../NEW_NN_1/NN_1.csv', index_col=0)
pred_new_nn_2 = pd.read_csv('../../NEW_NN_2/NN_2.csv', index_col=0)
pred_new_nn_3 = pd.read_csv('../../NEW_NN_3/NN_3.csv', index_col=0)
pred_new_nn_4 = pd.read_csv('../../NEW_NN_4/NN_4.csv', index_col=0)
pred_new_nn_3.loc[pred_new_nn_3['loss']>MAX_VALUE,:] = MAX_VALUE
pred_new_nn_4.loc[pred_new_nn_4['loss']>MAX_VALUE,:] = MAX_VALUE
pred_new_nn_1_65 = pd.read_csv('../../NEW_NN_1_65/NN_1.csv', index_col=0)

pred_xgb_1_retrain = pd.read_csv('../../XGB_1/XGB_retrain_1.csv', index_col=0)
pred_xgb_2_retrain = pd.read_csv('../../XGB_2/XGB_retrain_2.csv', index_col=0)
pred_xgb_3_retrain = pd.read_csv('../../XGB_3/XGB_retrain_3.csv', index_col=0)

pred_xgb_1 = pd.read_csv('../../XGB_1/XGB_1.csv', index_col=0)
pred_xgb_2 = pd.read_csv('../../XGB_2/XGB_2.csv', index_col=0)
pred_xgb_3 = pd.read_csv('../../XGB_3/XGB_3.csv', index_col=0)


print mean_absolute_error(train['loss'], pred_nn_1_retrain)
print mean_absolute_error(train['loss'], pred_nn_2_retrain)
print mean_absolute_error(train['loss'], pred_nn_3_retrain)
print mean_absolute_error(train['loss'], pred_nn_4_retrain)
print mean_absolute_error(train['loss'], pred_nn_5_retrain)
print mean_absolute_error(train['loss'], pred_nn_6_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_nn_1_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_2_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_3_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_4_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_5_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_6_fix_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_new_nn_1_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_2_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_3_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_4_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_new_nn_1_65_retrain)

df_retrain = pd.concat([
                           pred_nn_1_fix_retrain['loss'],    #1
                           pred_nn_2_fix_retrain['loss'],    #2
                           pred_nn_3_fix_retrain['loss'],    #3
                           pred_nn_4_fix_retrain['loss'],    #4
                           pred_nn_5_fix_retrain['loss'],    #5
                           pred_nn_6_fix_retrain['loss'],    #6
                           pred_new_nn_1_retrain['loss'],    #7
                           pred_new_nn_2_retrain['loss'],    #8
                           pred_new_nn_3_retrain['loss'],    #9
                           pred_new_nn_4_retrain['loss'],    #10
                           pred_new_nn_1_65_retrain['loss'], #11
                           pred_xgb_1_retrain['loss'],       #12
                           pred_xgb_2_retrain['loss'],       #13
                           pred_xgb_3_retrain['loss'],       #14
                           pred_nn_1_retrain['loss'],        #15
                           pred_nn_2_retrain['loss'],        #16
                           pred_nn_3_retrain['loss'],        #17
                           pred_nn_4_retrain['loss'],        #18
                           pred_nn_5_retrain['loss'],        #19
                           pred_nn_6_retrain['loss'],        #20
                           ], axis=1)


from scipy.optimize import minimize

# ======================== NN optimize ======================== #
def f(coord,args):
    pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,pred_11,pred_12,pred_13,pred_14,pred_15,pred_16,pred_17,pred_18,pred_19,pred_20,r = args
    return np.mean( np.abs(coord[0]*pred_1 + coord[1]*pred_2 + coord[2]*pred_3
                          +coord[3]*pred_4 + coord[4]*pred_5 + coord[5]*pred_6
                          +coord[6]*pred_7 + coord[7]*pred_5 + coord[8]*pred_6
                          +coord[9]*pred_10 + coord[10]*pred_5 + coord[11]*pred_6
                          +coord[12]*pred_13 + coord[13]*pred_5 + coord[14]*pred_6
                          +coord[15]*pred_16 + coord[16]*pred_5 + coord[17]*pred_6
                          +coord[18]*pred_19 + coord[19]*pred_20
                          - r))


initial_guess = np.array([0.5 for x in range(20)])


res = minimize(f,initial_guess,args = [
                                        pred_nn_1_retrain['loss'].values,
                                        pred_nn_2_retrain['loss'].values,
                                        pred_nn_3_retrain['loss'].values,
                                        pred_nn_4_retrain['loss'].values,
                                        pred_nn_5_retrain['loss'].values,
                                        pred_nn_6_retrain['loss'].values,
                           pred_new_nn_1_retrain['loss'].values,    #7
                           pred_new_nn_2_retrain['loss'].values,    #8
                           pred_new_nn_3_retrain['loss'].values,    #9
                           pred_new_nn_4_retrain['loss'].values,    #10
                           pred_new_nn_1_65_retrain['loss'].values, #11
                           pred_xgb_1_retrain['loss'].values,       #12
                           pred_xgb_2_retrain['loss'].values,       #13
                           pred_xgb_3_retrain['loss'].values,       #14
                           pred_nn_1_retrain['loss'].values,        #15
                           pred_nn_2_retrain['loss'].values,        #16
                           pred_nn_3_retrain['loss'].values,        #17
                           pred_nn_4_retrain['loss'].values,        #18
                           pred_nn_5_retrain['loss'].values,        #19
                           pred_nn_6_retrain['loss'].values,        #20
                                       train['loss'].values]
                              ,method='SLSQP')

print res
