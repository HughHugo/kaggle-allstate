# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('../../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
MIN_VALUE = np.min(train['loss'])

def bound_df_retrain(df):
    assert df.shape[0] == 188318
    assert df.columns[0] == 'loss'
    df.loc[df['loss']>MAX_VALUE,:] = MAX_VALUE
    df.loc[df['loss']<MIN_VALUE,:] = MIN_VALUE

### Train ###
pred_nn_1_fix_retrain = pd.read_csv('../../NN_1_fix/NN_retrain_1.csv', index_col=0)
bound_df_retrain(pred_nn_1_fix_retrain)
pred_nn_2_fix_retrain = pd.read_csv('../../NN_2_fix/NN_retrain_2.csv', index_col=0)
bound_df_retrain(pred_nn_2_fix_retrain)
pred_nn_3_fix_retrain = pd.read_csv('../../NN_3_fix/NN_retrain_3.csv', index_col=0)
bound_df_retrain(pred_nn_3_fix_retrain)
pred_nn_4_fix_retrain = pd.read_csv('../../NN_4_fix/NN_retrain_4.csv', index_col=0)
bound_df_retrain(pred_nn_4_fix_retrain)
pred_nn_5_fix_retrain = pd.read_csv('../../NN_5_fix/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_nn_5_fix_retrain)
pred_nn_6_fix_retrain = pd.read_csv('../../NN_6_fix/NN_retrain_6.csv', index_col=0)
bound_df_retrain(pred_nn_6_fix_retrain)


pred_new_nn_1_retrain = pd.read_csv('../../NEW_NN_1/NN_retrain_1.csv', index_col=0)
bound_df_retrain(pred_new_nn_1_retrain)
pred_new_nn_2_retrain = pd.read_csv('../../NEW_NN_2/NN_retrain_2.csv', index_col=0)
bound_df_retrain(pred_new_nn_2_retrain)
pred_new_nn_3_retrain = pd.read_csv('../../NEW_NN_3/NN_retrain_3.csv', index_col=0)
bound_df_retrain(pred_new_nn_3_retrain)
pred_new_nn_4_retrain = pd.read_csv('../../NEW_NN_4/NN_retrain_4.csv', index_col=0)
bound_df_retrain(pred_new_nn_4_retrain)
pred_new_nn_5_retrain = pd.read_csv('../../NEW_NN_5/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_retrain)
pred_new_nn_6_retrain = pd.read_csv('../../NEW_NN_6/NN_retrain_6.csv', index_col=0)
bound_df_retrain(pred_new_nn_6_retrain)
pred_new_nn_7_retrain = pd.read_csv('../../NEW_NN_7/NN_retrain_7.csv', index_col=0)
bound_df_retrain(pred_new_nn_7_retrain)


pred_new_nn_1_65_retrain = pd.read_csv('../../NEW_NN_1_65/NN_retrain_1.csv', index_col=0)
bound_df_retrain(pred_new_nn_1_65_retrain)
pred_new_nn_2_65_retrain = pd.read_csv('../../NEW_NN_2_65/NN_retrain_2.csv', index_col=0)
bound_df_retrain(pred_new_nn_2_65_retrain)


pred_xgb_1_retrain = pd.read_csv('../../XGB_1/XGB_retrain_1.csv', index_col=0)
bound_df_retrain(pred_xgb_1_retrain)
pred_xgb_2_retrain = pd.read_csv('../../XGB_2/XGB_retrain_2.csv', index_col=0)
bound_df_retrain(pred_xgb_2_retrain)
pred_xgb_3_retrain = pd.read_csv('../../XGB_3/XGB_retrain_3.csv', index_col=0)
bound_df_retrain(pred_xgb_3_retrain)
pred_xgb_6_retrain = pd.read_csv('../../XGB_6/XGB_retrain_6.csv', index_col=0)
bound_df_retrain(pred_xgb_6_retrain)
pred_xgb_9_retrain = pd.read_csv('../../XGB_9/XGB_retrain_9.csv', index_col=0)
bound_df_retrain(pred_xgb_9_retrain)


def bound_df_test(df):
    assert df.shape[0] == 125546
    assert df.columns[0] == 'loss'
    df.loc[df['loss']>MAX_VALUE,:] = MAX_VALUE
    df.loc[df['loss']<MIN_VALUE,:] = MIN_VALUE

### Test ###
pred_nn_1_fix = pd.read_csv('../../NN_1_fix/NN_1.csv', index_col=0)
bound_df_test(pred_nn_1_fix)
pred_nn_2_fix = pd.read_csv('../../NN_2_fix/NN_2.csv', index_col=0)
bound_df_test(pred_nn_2_fix)
pred_nn_3_fix = pd.read_csv('../../NN_3_fix/NN_3.csv', index_col=0)
bound_df_test(pred_nn_3_fix)
pred_nn_4_fix = pd.read_csv('../../NN_4_fix/NN_4.csv', index_col=0)
bound_df_test(pred_nn_4_fix)
pred_nn_5_fix = pd.read_csv('../../NN_5_fix/NN_5.csv', index_col=0)
bound_df_test(pred_nn_5_fix)
pred_nn_6_fix = pd.read_csv('../../NN_6_fix/NN_6.csv', index_col=0)
bound_df_test(pred_nn_6_fix)


pred_new_nn_1 = pd.read_csv('../../NEW_NN_1/NN_1.csv', index_col=0)
bound_df_test(pred_new_nn_1)
pred_new_nn_2 = pd.read_csv('../../NEW_NN_2/NN_2.csv', index_col=0)
bound_df_test(pred_new_nn_2)
pred_new_nn_3 = pd.read_csv('../../NEW_NN_3/NN_3.csv', index_col=0)
bound_df_test(pred_new_nn_3)
pred_new_nn_4 = pd.read_csv('../../NEW_NN_4/NN_4.csv', index_col=0)
bound_df_test(pred_new_nn_4)
pred_new_nn_5 = pd.read_csv('../../NEW_NN_5/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5)
pred_new_nn_6 = pd.read_csv('../../NEW_NN_6/NN_6.csv', index_col=0)
bound_df_test(pred_new_nn_6)
pred_new_nn_7 = pd.read_csv('../../NEW_NN_7/NN_7.csv', index_col=0)
bound_df_test(pred_new_nn_7)


pred_new_nn_1_65 = pd.read_csv('../../NEW_NN_1_65/NN_1.csv', index_col=0)
bound_df_test(pred_new_nn_1_65)
pred_new_nn_2_65 = pd.read_csv('../../NEW_NN_2_65/NN_2.csv', index_col=0)
bound_df_test(pred_new_nn_2_65)


pred_xgb_1 = pd.read_csv('../../XGB_1/XGB_1.csv', index_col=0)
bound_df_test(pred_xgb_1)
pred_xgb_2 = pd.read_csv('../../XGB_2/XGB_2.csv', index_col=0)
bound_df_test(pred_xgb_2)
pred_xgb_3 = pd.read_csv('../../XGB_3/XGB_3.csv', index_col=0)
bound_df_test(pred_xgb_3)
pred_xgb_6 = pd.read_csv('../../XGB_6/XGB_6.csv', index_col=0)
bound_df_test(pred_xgb_6)
pred_xgb_9 = pd.read_csv('../../XGB_9/XGB_9.csv', index_col=0)
bound_df_test(pred_xgb_9)


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
print mean_absolute_error(train['loss'], pred_new_nn_5_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_6_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_7_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_new_nn_1_65_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_2_65_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_xgb_1_retrain)
print mean_absolute_error(train['loss'], pred_xgb_2_retrain)
print mean_absolute_error(train['loss'], pred_xgb_3_retrain)
print mean_absolute_error(train['loss'], pred_xgb_6_retrain)
print mean_absolute_error(train['loss'], pred_xgb_9_retrain)


# ======================== optimize ======================== #
from scipy.optimize import minimize

args = [
    pred_nn_1_fix_retrain['loss'].values,       #1
    pred_nn_2_fix_retrain['loss'].values,       #2
    pred_nn_3_fix_retrain['loss'].values,       #3
    pred_nn_4_fix_retrain['loss'].values,       #4
    pred_nn_5_fix_retrain['loss'].values,       #5
    pred_nn_6_fix_retrain['loss'].values,       #6
    pred_new_nn_1_retrain['loss'].values,       #7
    pred_new_nn_2_retrain['loss'].values,       #8
    pred_new_nn_3_retrain['loss'].values,       #9
    pred_new_nn_4_retrain['loss'].values,       #10
    pred_new_nn_5_retrain['loss'].values,       #11
    pred_new_nn_6_retrain['loss'].values,       #12
    pred_new_nn_7_retrain['loss'].values,       #13
    pred_new_nn_1_65_retrain['loss'].values,    #14
    pred_new_nn_2_65_retrain['loss'].values,    #15
    pred_xgb_1_retrain['loss'].values,          #16
    pred_xgb_2_retrain['loss'].values,          #17
    pred_xgb_3_retrain['loss'].values,          #18
    pred_xgb_6_retrain['loss'].values,          #19
    pred_xgb_9_retrain['loss'].values,          #20
    train['loss'].values
]

print len(args)-1
pe= 5

def f(coord,args):
    return np.mean( np.abs(coord[pe*0]*args[0] + coord[pe*0+1]*(args[0] ** 2) + coord[pe*0+2]*np.log(args[0]) + coord[pe*0+3]*1/(1.0+args[0]) + coord[pe*0+4]*(args[0] ** 0.5)
     + coord[pe*1]*args[1] + coord[pe*1+1]*(args[1] ** 2) + coord[pe*1+2]*np.log(args[1]) + coord[pe*1+3]*1/(1.0+args[1]) + coord[pe*1+4]*(args[1] ** 0.5)
     + coord[pe*2]*args[2] + coord[pe*2+1]*(args[2] ** 2) + coord[pe*2+2]*np.log(args[2]) + coord[pe*2+3]*1/(1.0+args[2]) + coord[pe*2+4]*(args[2] ** 0.5)
     + coord[pe*3]*args[3] + coord[pe*3+1]*(args[3] ** 2) + coord[pe*3+2]*np.log(args[3]) + coord[pe*3+3]*1/(1.0+args[3]) + coord[pe*3+4]*(args[3] ** 0.5)
     + coord[pe*4]*args[4] + coord[pe*4+1]*(args[4] ** 2) + coord[pe*4+2]*np.log(args[4]) + coord[pe*4+3]*1/(1.0+args[4]) + coord[pe*4+4]*(args[4] ** 0.5)
     + coord[pe*5]*args[5] + coord[pe*5+1]*(args[5] ** 2) + coord[pe*5+2]*np.log(args[5]) + coord[pe*5+3]*1/(1.0+args[5]) + coord[pe*5+4]*(args[5] ** 0.5)
     + coord[pe*6]*args[6] + coord[pe*6+1]*(args[6] ** 2) + coord[pe*6+2]*np.log(args[6]) + coord[pe*6+3]*1/(1.0+args[6]) + coord[pe*6+4]*(args[6] ** 0.5)
     + coord[pe*7]*args[7] + coord[pe*7+1]*(args[7] ** 2) + coord[pe*7+2]*np.log(args[7]) + coord[pe*7+3]*1/(1.0+args[7]) + coord[pe*7+4]*(args[7] ** 0.5)
     + coord[pe*8]*args[8] + coord[pe*8+1]*(args[8] ** 2) + coord[pe*8+2]*np.log(args[8]) + coord[pe*8+3]*1/(1.0+args[8]) + coord[pe*8+4]*(args[8] ** 0.5)
     + coord[pe*9]*args[9] + coord[pe*9+1]*(args[9] ** 2) + coord[pe*9+2]*np.log(args[9]) + coord[pe*9+3]*1/(1.0+args[9]) + coord[pe*9+4]*(args[9] ** 0.5)
     + coord[pe*10]*args[10] + coord[pe*10+1]*(args[10] ** 2) + coord[pe*10+2]*np.log(args[10]) + coord[pe*10+3]*1/(1.0+args[10]) + coord[pe*10+4]*(args[10] ** 0.5)
     + coord[pe*11]*args[11] + coord[pe*11+1]*(args[11] ** 2) + coord[pe*11+2]*np.log(args[11]) + coord[pe*11+3]*1/(1.0+args[11]) + coord[pe*11+4]*(args[11] ** 0.5)
     + coord[pe*12]*args[12] + coord[pe*12+1]*(args[12] ** 2) + coord[pe*12+2]*np.log(args[12]) + coord[pe*12+3]*1/(1.0+args[12]) + coord[pe*12+4]*(args[12] ** 0.5)
     + coord[pe*13]*args[13] + coord[pe*13+1]*(args[13] ** 2) + coord[pe*13+2]*np.log(args[13]) + coord[pe*13+3]*1/(1.0+args[13]) + coord[pe*13+4]*(args[13] ** 0.5)
     + coord[pe*14]*args[14] + coord[pe*14+1]*(args[14] ** 2) + coord[pe*14+2]*np.log(args[14]) + coord[pe*14+3]*1/(1.0+args[14]) + coord[pe*14+4]*(args[14] ** 0.5)
     + coord[pe*15]*args[15] + coord[pe*15+1]*(args[15] ** 2) + coord[pe*15+2]*np.log(args[15]) + coord[pe*15+3]*1/(1.0+args[15]) + coord[pe*15+4]*(args[15] ** 0.5)
     + coord[pe*16]*args[16] + coord[pe*16+1]*(args[16] ** 2) + coord[pe*16+2]*np.log(args[16]) + coord[pe*16+3]*1/(1.0+args[16]) + coord[pe*16+4]*(args[16] ** 0.5)
     + coord[pe*17]*args[17] + coord[pe*17+1]*(args[17] ** 2) + coord[pe*17+2]*np.log(args[17]) + coord[pe*17+3]*1/(1.0+args[17]) + coord[pe*17+4]*(args[17] ** 0.5)
     + coord[pe*18]*args[18] + coord[pe*18+1]*(args[18] ** 2) + coord[pe*18+2]*np.log(args[18]) + coord[pe*18+3]*1/(1.0+args[18]) + coord[pe*18+4]*(args[18] ** 0.5)
     + coord[pe*19]*args[19] + coord[pe*19+1]*(args[19] ** 2) + coord[pe*19+2]*np.log(args[19]) + coord[pe*19+3]*1/(1.0+args[19]) + coord[pe*19+4]*(args[19] ** 0.5)
     + coord[pe*20+0]*(args[19] - args[10])
     + coord[pe*20+1]*((args[19]-args[10]) ** 2)
     + coord[pe*20+2]*np.log(1+abs(args[19]-args[10]))
     + coord[pe*20+3]*1/(1.0+args[19]-args[10])
     + coord[pe*20+4]*(abs(args[19]-args[10]) ** 0.5)
     + coord[pe*20+5]
     + coord[pe*20+6]*(args[19] - args[17])
     + coord[pe*20+7]*((args[19]-args[17]) ** 2)
     + coord[pe*20+8]*np.log(1+abs(args[19]-args[17]))
     + coord[pe*20+9]*1/(1.0+abs(args[19]-args[17]))
     + coord[pe*20+10]*(abs(args[19]-args[17]) ** 0.5)
     + coord[pe*20+11]*(args[10] - args[17])
     + coord[pe*20+12]*((args[10]-args[17]) ** 2)
     + coord[pe*20+13]*np.log(1+abs(args[10]-args[17]))
     + coord[pe*20+14]*1/(1.0+args[10]-args[17])
     + coord[pe*20+15]*(abs(args[10]-args[17]) ** 0.5)
     - args[-1]) )


initial_guess = np.array([0.1 for x in range(pe * 20 + 6 + 5 + 5)])

Nfeval = 1
def callbackF(Xi):
    global Nfeval
    print '{0:4d}   {1: 3.6f}'.format(Nfeval, f(Xi, args))
    Nfeval += 1


res = minimize(f,initial_guess, args = args
                              ,method='SLSQP'
                              ,options={"maxiter":1000000,"disp":True}
                              ,callback=callbackF)

print res

pred_ensemble = (res.x[pe*0]*args[0] + res.x[pe*0+1]*(args[0] ** 2) + res.x[pe*0+2]*np.log(args[0]) + res.x[pe*0+3]*1/(1.0+args[0]) + res.x[pe*0+4]*(args[0] ** 0.5)
 + res.x[pe*1]*args[1] + res.x[pe*1+1]*(args[1] ** 2) + res.x[pe*1+2]*np.log(args[1]) + res.x[pe*1+3]*1/(1.0+args[1]) + res.x[pe*1+4]*(args[1] ** 0.5)
 + res.x[pe*2]*args[2] + res.x[pe*2+1]*(args[2] ** 2) + res.x[pe*2+2]*np.log(args[2]) + res.x[pe*2+3]*1/(1.0+args[2]) + res.x[pe*2+4]*(args[2] ** 0.5)
 + res.x[pe*3]*args[3] + res.x[pe*3+1]*(args[3] ** 2) + res.x[pe*3+2]*np.log(args[3]) + res.x[pe*3+3]*1/(1.0+args[3]) + res.x[pe*3+4]*(args[3] ** 0.5)
 + res.x[pe*4]*args[4] + res.x[pe*4+1]*(args[4] ** 2) + res.x[pe*4+2]*np.log(args[4]) + res.x[pe*4+3]*1/(1.0+args[4]) + res.x[pe*4+4]*(args[4] ** 0.5)
 + res.x[pe*5]*args[5] + res.x[pe*5+1]*(args[5] ** 2) + res.x[pe*5+2]*np.log(args[5]) + res.x[pe*5+3]*1/(1.0+args[5]) + res.x[pe*5+4]*(args[5] ** 0.5)
 + res.x[pe*6]*args[6] + res.x[pe*6+1]*(args[6] ** 2) + res.x[pe*6+2]*np.log(args[6]) + res.x[pe*6+3]*1/(1.0+args[6]) + res.x[pe*6+4]*(args[6] ** 0.5)
 + res.x[pe*7]*args[7] + res.x[pe*7+1]*(args[7] ** 2) + res.x[pe*7+2]*np.log(args[7]) + res.x[pe*7+3]*1/(1.0+args[7]) + res.x[pe*7+4]*(args[7] ** 0.5)
 + res.x[pe*8]*args[8] + res.x[pe*8+1]*(args[8] ** 2) + res.x[pe*8+2]*np.log(args[8]) + res.x[pe*8+3]*1/(1.0+args[8]) + res.x[pe*8+4]*(args[8] ** 0.5)
 + res.x[pe*9]*args[9] + res.x[pe*9+1]*(args[9] ** 2) + res.x[pe*9+2]*np.log(args[9]) + res.x[pe*9+3]*1/(1.0+args[9]) + res.x[pe*9+4]*(args[9] ** 0.5)
 + res.x[pe*10]*args[10] + res.x[pe*10+1]*(args[10] ** 2) + res.x[pe*10+2]*np.log(args[10]) + res.x[pe*10+3]*1/(1.0+args[10]) + res.x[pe*10+4]*(args[10] ** 0.5)
 + res.x[pe*11]*args[11] + res.x[pe*11+1]*(args[11] ** 2) + res.x[pe*11+2]*np.log(args[11]) + res.x[pe*11+3]*1/(1.0+args[11]) + res.x[pe*11+4]*(args[11] ** 0.5)
 + res.x[pe*12]*args[12] + res.x[pe*12+1]*(args[12] ** 2) + res.x[pe*12+2]*np.log(args[12]) + res.x[pe*12+3]*1/(1.0+args[12]) + res.x[pe*12+4]*(args[12] ** 0.5)
 + res.x[pe*13]*args[13] + res.x[pe*13+1]*(args[13] ** 2) + res.x[pe*13+2]*np.log(args[13]) + res.x[pe*13+3]*1/(1.0+args[13]) + res.x[pe*13+4]*(args[13] ** 0.5)
 + res.x[pe*14]*args[14] + res.x[pe*14+1]*(args[14] ** 2) + res.x[pe*14+2]*np.log(args[14]) + res.x[pe*14+3]*1/(1.0+args[14]) + res.x[pe*14+4]*(args[14] ** 0.5)
 + res.x[pe*15]*args[15] + res.x[pe*15+1]*(args[15] ** 2) + res.x[pe*15+2]*np.log(args[15]) + res.x[pe*15+3]*1/(1.0+args[15]) + res.x[pe*15+4]*(args[15] ** 0.5)
 + res.x[pe*16]*args[16] + res.x[pe*16+1]*(args[16] ** 2) + res.x[pe*16+2]*np.log(args[16]) + res.x[pe*16+3]*1/(1.0+args[16]) + res.x[pe*16+4]*(args[16] ** 0.5)
 + res.x[pe*17]*args[17] + res.x[pe*17+1]*(args[17] ** 2) + res.x[pe*17+2]*np.log(args[17]) + res.x[pe*17+3]*1/(1.0+args[17]) + res.x[pe*17+4]*(args[17] ** 0.5)
 + res.x[pe*18]*args[18] + res.x[pe*18+1]*(args[18] ** 2) + res.x[pe*18+2]*np.log(args[18]) + res.x[pe*18+3]*1/(1.0+args[18]) + res.x[pe*18+4]*(args[18] ** 0.5)
 + res.x[pe*19]*args[19] + res.x[pe*19+1]*(args[19] ** 2) + res.x[pe*19+2]*np.log(args[19]) + res.x[pe*19+3]*1/(1.0+args[19]) + res.x[pe*19+4]*(args[19] ** 0.5)
 + res.x[pe*20]*args[20] + res.x[pe*20+1]*(args[20] ** 2) + res.x[pe*20+2]*np.log(args[20]) + res.x[pe*20+3]*1/(1.0+args[20]) + res.x[pe*20+4]*(args[20] ** 0.5)
 + res.x[pe*21]*args[21] + res.x[pe*21+1]*(args[21] ** 2) + res.x[pe*21+2]*np.log(args[21]) + res.x[pe*21+3]*1/(1.0+args[21]) + res.x[pe*21+4]*(args[21] ** 0.5)
 + res.x[pe*22]*args[22] + res.x[pe*22+1]*(args[22] ** 2) + res.x[pe*22+2]*np.log(args[22]) + res.x[pe*22+3]*1/(1.0+args[22]) + res.x[pe*22+4]*(args[22] ** 0.5)
 + res.x[pe*23]*args[23] + res.x[pe*23+1]*(args[23] ** 2) + res.x[pe*23+2]*np.log(args[23]) + res.x[pe*23+3]*1/(1.0+args[23]) + res.x[pe*23+4]*(args[23] ** 0.5)
 + res.x[pe*24]*args[24] + res.x[pe*24+1]*(args[24] ** 2) + res.x[pe*24+2]*np.log(args[24]) + res.x[pe*24+3]*1/(1.0+args[24]) + res.x[pe*24+4]*(args[24] ** 0.5)
 + res.x[pe*25]*args[25] + res.x[pe*25+1]*(args[25] ** 2) + res.x[pe*25+2]*np.log(args[25]) + res.x[pe*25+3]*1/(1.0+args[25]) + res.x[pe*25+4]*(args[25] ** 0.5)
 + res.x[pe*26]*args[26] + res.x[pe*26+1]*(args[26] ** 2) + res.x[pe*26+2]*np.log(args[26]) + res.x[pe*26+3]*1/(1.0+args[26]) + res.x[pe*26+4]*(args[26] ** 0.5)
 + res.x[pe*27]*args[27] + res.x[pe*27+1]*(args[27] ** 2) + res.x[pe*27+2]*np.log(args[27]) + res.x[pe*27+3]*1/(1.0+args[27]) + res.x[pe*27+4]*(args[27] ** 0.5)
 + res.x[pe*28]*args[28] + res.x[pe*28+1]*(args[28] ** 2) + res.x[pe*28+2]*np.log(args[28]) + res.x[pe*28+3]*1/(1.0+args[28]) + res.x[pe*28+4]*(args[28] ** 0.5)
 + res.x[pe*29+0]*(args[23] - args[16])
 + res.x[pe*29+1]*((args[23]-args[16]) ** 2)
 + res.x[pe*29+2]*np.log(abs(args[23]-args[16]))
 + res.x[pe*29+3]*1/(1.0+args[23]-args[16])
 + res.x[pe*29+4]*(abs(args[23]-args[16]) ** 0.5)
 + res.x[pe*29+5]
 + res.x[pe*29+6]*(args[25] - args[16])
 + res.x[pe*29+7]*((args[25]-args[16]) ** 2)
 + res.x[pe*29+8]*np.log(abs(args[25]-args[16]))
 + res.x[pe*29+9]*1/(1.0+args[25]-args[16])
 + res.x[pe*29+10]*(abs(args[25]-args[16]) ** 0.5)
 + res.x[pe*29+6]*(args[25] - args[23])
 + res.x[pe*29+7]*((args[25]-args[23]) ** 2)
 + res.x[pe*29+8]*np.log(1+abs(args[25]-args[23]))
 + res.x[pe*29+9]*1/(1.0+abs(args[25]-args[23]))
 + res.x[pe*29+10]*(abs(args[25]-args[23]) ** 0.5)
)

pred_ensemble = pd.DataFrame(pred_ensemble)
pred_ensemble.columns = ['loss']
bound_df_retrain(pred_ensemble)
pred_ensemble.to_csv("retrain.csv")


####### prediction
args = [
    pred_nn_1['loss'],           #1
    pred_nn_2['loss'],           #2
    pred_nn_3['loss'],           #3
    pred_nn_4['loss'],           #4
    pred_nn_5['loss'],           #5
    pred_nn_6['loss'],           #6
    pred_nn_1_fix['loss'],       #7
    pred_nn_2_fix['loss'],       #8
    pred_nn_3_fix['loss'],       #9
    pred_nn_4_fix['loss'],       #10
    pred_nn_5_fix['loss'],       #11
    pred_nn_6_fix['loss'],       #12
    pred_new_nn_1['loss'],       #13
    pred_new_nn_2['loss'],       #14
    pred_new_nn_3['loss'],       #15
    pred_new_nn_4['loss'],       #16
    pred_new_nn_5['loss'],       #17
    pred_new_nn_6['loss'],       #18
    pred_new_nn_7['loss'],       #19
    pred_new_nn_1_65['loss'],    #20
    pred_new_nn_2_65['loss'],    #21
    pred_xgb_1['loss'],          #22
    pred_xgb_2['loss'],          #23
    pred_xgb_3['loss'],          #24
    pred_xgb_6['loss'],          #25
    pred_xgb_9['loss'],          #26
    pred_new_nn_2_LEAK['loss'],  #27
    pred_new_nn_7_LEAK['loss'],  #28
    pred_xgb_9_LEAK['loss'],     #29
]

pred_ensemble = (res.x[pe*0]*args[0] + res.x[pe*0+1]*(args[0] ** 2) + res.x[pe*0+2]*np.log(args[0]) + res.x[pe*0+3]*1/(1.0+args[0]) + res.x[pe*0+4]*(args[0] ** 0.5)
 + res.x[pe*1]*args[1] + res.x[pe*1+1]*(args[1] ** 2) + res.x[pe*1+2]*np.log(args[1]) + res.x[pe*1+3]*1/(1.0+args[1]) + res.x[pe*1+4]*(args[1] ** 0.5)
 + res.x[pe*2]*args[2] + res.x[pe*2+1]*(args[2] ** 2) + res.x[pe*2+2]*np.log(args[2]) + res.x[pe*2+3]*1/(1.0+args[2]) + res.x[pe*2+4]*(args[2] ** 0.5)
 + res.x[pe*3]*args[3] + res.x[pe*3+1]*(args[3] ** 2) + res.x[pe*3+2]*np.log(args[3]) + res.x[pe*3+3]*1/(1.0+args[3]) + res.x[pe*3+4]*(args[3] ** 0.5)
 + res.x[pe*4]*args[4] + res.x[pe*4+1]*(args[4] ** 2) + res.x[pe*4+2]*np.log(args[4]) + res.x[pe*4+3]*1/(1.0+args[4]) + res.x[pe*4+4]*(args[4] ** 0.5)
 + res.x[pe*5]*args[5] + res.x[pe*5+1]*(args[5] ** 2) + res.x[pe*5+2]*np.log(args[5]) + res.x[pe*5+3]*1/(1.0+args[5]) + res.x[pe*5+4]*(args[5] ** 0.5)
 + res.x[pe*6]*args[6] + res.x[pe*6+1]*(args[6] ** 2) + res.x[pe*6+2]*np.log(args[6]) + res.x[pe*6+3]*1/(1.0+args[6]) + res.x[pe*6+4]*(args[6] ** 0.5)
 + res.x[pe*7]*args[7] + res.x[pe*7+1]*(args[7] ** 2) + res.x[pe*7+2]*np.log(args[7]) + res.x[pe*7+3]*1/(1.0+args[7]) + res.x[pe*7+4]*(args[7] ** 0.5)
 + res.x[pe*8]*args[8] + res.x[pe*8+1]*(args[8] ** 2) + res.x[pe*8+2]*np.log(args[8]) + res.x[pe*8+3]*1/(1.0+args[8]) + res.x[pe*8+4]*(args[8] ** 0.5)
 + res.x[pe*9]*args[9] + res.x[pe*9+1]*(args[9] ** 2) + res.x[pe*9+2]*np.log(args[9]) + res.x[pe*9+3]*1/(1.0+args[9]) + res.x[pe*9+4]*(args[9] ** 0.5)
 + res.x[pe*10]*args[10] + res.x[pe*10+1]*(args[10] ** 2) + res.x[pe*10+2]*np.log(args[10]) + res.x[pe*10+3]*1/(1.0+args[10]) + res.x[pe*10+4]*(args[10] ** 0.5)
 + res.x[pe*11]*args[11] + res.x[pe*11+1]*(args[11] ** 2) + res.x[pe*11+2]*np.log(args[11]) + res.x[pe*11+3]*1/(1.0+args[11]) + res.x[pe*11+4]*(args[11] ** 0.5)
 + res.x[pe*12]*args[12] + res.x[pe*12+1]*(args[12] ** 2) + res.x[pe*12+2]*np.log(args[12]) + res.x[pe*12+3]*1/(1.0+args[12]) + res.x[pe*12+4]*(args[12] ** 0.5)
 + res.x[pe*13]*args[13] + res.x[pe*13+1]*(args[13] ** 2) + res.x[pe*13+2]*np.log(args[13]) + res.x[pe*13+3]*1/(1.0+args[13]) + res.x[pe*13+4]*(args[13] ** 0.5)
 + res.x[pe*14]*args[14] + res.x[pe*14+1]*(args[14] ** 2) + res.x[pe*14+2]*np.log(args[14]) + res.x[pe*14+3]*1/(1.0+args[14]) + res.x[pe*14+4]*(args[14] ** 0.5)
 + res.x[pe*15]*args[15] + res.x[pe*15+1]*(args[15] ** 2) + res.x[pe*15+2]*np.log(args[15]) + res.x[pe*15+3]*1/(1.0+args[15]) + res.x[pe*15+4]*(args[15] ** 0.5)
 + res.x[pe*16]*args[16] + res.x[pe*16+1]*(args[16] ** 2) + res.x[pe*16+2]*np.log(args[16]) + res.x[pe*16+3]*1/(1.0+args[16]) + res.x[pe*16+4]*(args[16] ** 0.5)
 + res.x[pe*17]*args[17] + res.x[pe*17+1]*(args[17] ** 2) + res.x[pe*17+2]*np.log(args[17]) + res.x[pe*17+3]*1/(1.0+args[17]) + res.x[pe*17+4]*(args[17] ** 0.5)
 + res.x[pe*18]*args[18] + res.x[pe*18+1]*(args[18] ** 2) + res.x[pe*18+2]*np.log(args[18]) + res.x[pe*18+3]*1/(1.0+args[18]) + res.x[pe*18+4]*(args[18] ** 0.5)
 + res.x[pe*19]*args[19] + res.x[pe*19+1]*(args[19] ** 2) + res.x[pe*19+2]*np.log(args[19]) + res.x[pe*19+3]*1/(1.0+args[19]) + res.x[pe*19+4]*(args[19] ** 0.5)
 + res.x[pe*20]*args[20] + res.x[pe*20+1]*(args[20] ** 2) + res.x[pe*20+2]*np.log(args[20]) + res.x[pe*20+3]*1/(1.0+args[20]) + res.x[pe*20+4]*(args[20] ** 0.5)
 + res.x[pe*21]*args[21] + res.x[pe*21+1]*(args[21] ** 2) + res.x[pe*21+2]*np.log(args[21]) + res.x[pe*21+3]*1/(1.0+args[21]) + res.x[pe*21+4]*(args[21] ** 0.5)
 + res.x[pe*22]*args[22] + res.x[pe*22+1]*(args[22] ** 2) + res.x[pe*22+2]*np.log(args[22]) + res.x[pe*22+3]*1/(1.0+args[22]) + res.x[pe*22+4]*(args[22] ** 0.5)
 + res.x[pe*23]*args[23] + res.x[pe*23+1]*(args[23] ** 2) + res.x[pe*23+2]*np.log(args[23]) + res.x[pe*23+3]*1/(1.0+args[23]) + res.x[pe*23+4]*(args[23] ** 0.5)
 + res.x[pe*24]*args[24] + res.x[pe*24+1]*(args[24] ** 2) + res.x[pe*24+2]*np.log(args[24]) + res.x[pe*24+3]*1/(1.0+args[24]) + res.x[pe*24+4]*(args[24] ** 0.5)
 + res.x[pe*25]*args[25] + res.x[pe*25+1]*(args[25] ** 2) + res.x[pe*25+2]*np.log(args[25]) + res.x[pe*25+3]*1/(1.0+args[25]) + res.x[pe*25+4]*(args[25] ** 0.5)
 + res.x[pe*26]*args[26] + res.x[pe*26+1]*(args[26] ** 2) + res.x[pe*26+2]*np.log(args[26]) + res.x[pe*26+3]*1/(1.0+args[26]) + res.x[pe*26+4]*(args[26] ** 0.5)
 + res.x[pe*27]*args[27] + res.x[pe*27+1]*(args[27] ** 2) + res.x[pe*27+2]*np.log(args[27]) + res.x[pe*27+3]*1/(1.0+args[27]) + res.x[pe*27+4]*(args[27] ** 0.5)
 + res.x[pe*28]*args[28] + res.x[pe*28+1]*(args[28] ** 2) + res.x[pe*28+2]*np.log(args[28]) + res.x[pe*28+3]*1/(1.0+args[28]) + res.x[pe*28+4]*(args[28] ** 0.5)
 + res.x[pe*29+0]*(args[23] - args[16])
 + res.x[pe*29+1]*((args[23]-args[16]) ** 2)
 + res.x[pe*29+2]*np.log(abs(args[23]-args[16]))
 + res.x[pe*29+3]*1/(1.0+args[23]-args[16])
 + res.x[pe*29+4]*(abs(args[23]-args[16]) ** 0.5)
 + res.x[pe*29+5]
 + res.x[pe*29+6]*(args[25] - args[16])
 + res.x[pe*29+7]*((args[25]-args[16]) ** 2)
 + res.x[pe*29+8]*np.log(abs(args[25]-args[16]))
 + res.x[pe*29+9]*1/(1.0+args[25]-args[16])
 + res.x[pe*29+10]*(abs(args[25]-args[16]) ** 0.5)
 + res.x[pe*29+11]*(args[25] - args[23])
 + res.x[pe*29+12]*((args[25]-args[23]) ** 2)
 + res.x[pe*29+13]*np.log(1+abs(args[25]-args[23]))
 + res.x[pe*29+14]*1/(1.0+abs(args[25]-args[23]))
 + res.x[pe*29+15]*(abs(args[25]-args[23]) ** 0.5)
)


pred_ensemble = pd.DataFrame(pred_ensemble)
pred_ensemble.columns = ['loss']
bound_df_test(pred_ensemble)
pred_ensemble.to_csv("pred_retrain.csv", index_label='id')