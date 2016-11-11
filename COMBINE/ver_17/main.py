# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

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
pred_new_nn_3_retrain = pd.read_csv('../../NEW_NN_3/NN_retrain_3.csv', index_col=0)
pred_new_nn_3_retrain.loc[pred_new_nn_3_retrain['loss']>40000,:] = np.median(pred_new_nn_3_retrain['loss'])
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
pred_new_nn_3.loc[pred_new_nn_3['loss']>40000,:] = np.median(pred_new_nn_3['loss'])

pred_xgb_1_retrain = pd.read_csv('../../XGB_1/XGB_retrain_1.csv', index_col=0)
pred_xgb_2_retrain = pd.read_csv('../../XGB_2/XGB_retrain_2.csv', index_col=0)
pred_xgb_3_retrain = pd.read_csv('../../XGB_3/XGB_retrain_3.csv', index_col=0)

pred_xgb_1 = pd.read_csv('../../XGB_1/XGB_1.csv', index_col=0)
pred_xgb_2 = pd.read_csv('../../XGB_2/XGB_2.csv', index_col=0)
pred_xgb_3 = pd.read_csv('../../XGB_3/XGB_3.csv', index_col=0)

train = pd.read_csv('../../input/train.csv', index_col=0)


print mean_absolute_error(train['loss'], pred_nn_1_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_2_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_3_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_4_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_5_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_6_fix_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_1_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_2_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_3_retrain)


# 3
from scipy.optimize import minimize

# ======================== NN optimize ======================== #
pe= 5
def f(coord,args):
    #pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,pred_11,pred_12,pred_13,pred_14,pred_15,pred_16,pred_17,pred_18,r = args
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
                         - args[-1]) )


initial_guess = np.array([0.1 for x in range(pe * 14)])


res = minimize(f,initial_guess,args = [
                                       pred_nn_1_fix_retrain['loss'].values, #1
                                       pred_nn_2_fix_retrain['loss'].values, #2
                                       pred_nn_3_fix_retrain['loss'].values, #3
                                       pred_nn_4_fix_retrain['loss'].values, #4
                                       pred_nn_5_fix_retrain['loss'].values, #5
                                       pred_nn_6_fix_retrain['loss'].values, #6
                                       pred_new_nn_1_retrain['loss'].values, #7
                                       pred_new_nn_3_retrain['loss'].values, #8
                                       pred_nn_1_retrain['loss'].values,     #9
                                       pred_nn_2_retrain['loss'].values,     #10
                                       pred_nn_3_retrain['loss'].values,     #11
                                       pred_nn_4_retrain['loss'].values,     #12
                                       pred_nn_5_retrain['loss'].values,     #13
                                       pred_nn_6_retrain['loss'].values,     #14
                                       train['loss'].values
                                       ]
                              ,method='SLSQP', options={"maxiter":1000000,"disp":True})

print res



pred_nn_test_list = [
                                       pred_nn_1_fix['loss'], #1
                                       pred_nn_2_fix['loss'], #2
                                       pred_nn_3_fix['loss'], #3
                                       pred_nn_4_fix['loss'], #4
                                       pred_nn_5_fix['loss'], #5
                                       pred_nn_6_fix['loss'], #6
                                       pred_new_nn_1['loss'], #7
                                       pred_new_nn_3['loss'], #8
                                       pred_nn_1['loss'],     #9
                                       pred_nn_2['loss'],     #10
                                       pred_nn_3['loss'],     #11
                                       pred_nn_4['loss'],     #12
                                       pred_nn_5['loss'],     #13
                                       pred_nn_6['loss'],     #14
        ]

pred_nn_retrain_list = [
                                       pred_nn_1_fix_retrain['loss'], #1
                                       pred_nn_2_fix_retrain['loss'], #2
                                       pred_nn_3_fix_retrain['loss'], #3
                                       pred_nn_4_fix_retrain['loss'], #4
                                       pred_nn_5_fix_retrain['loss'], #5
                                       pred_nn_6_fix_retrain['loss'], #6
                                       pred_new_nn_1_retrain['loss'], #7
                                       pred_new_nn_3_retrain['loss'], #8
                                       pred_nn_1_retrain['loss'],     #9
                                       pred_nn_2_retrain['loss'],     #10
                                       pred_nn_3_retrain['loss'],     #11
                                       pred_nn_4_retrain['loss'],     #12
                                       pred_nn_5_retrain['loss'],     #13
                                       pred_nn_6_retrain['loss'],     #14
        ]

pred_nn_retrain = (res.x[pe*0]*pred_nn_retrain_list[0] + res.x[pe*0+1]*(pred_nn_retrain_list[0] ** 2) + res.x[pe*0+2]*np.log(pred_nn_retrain_list[0]) + res.x[pe*0+3]*1/(1.0+pred_nn_retrain_list[0]) + res.x[pe*0+4]*(pred_nn_retrain_list[0] ** 0.5)
              + res.x[pe*1]*pred_nn_retrain_list[1] + res.x[pe*1+1]*(pred_nn_retrain_list[1] ** 2) + res.x[pe*1+2]*np.log(pred_nn_retrain_list[1]) + res.x[pe*1+3]*1/(1.0+pred_nn_retrain_list[1]) + res.x[pe*1+4]*(pred_nn_retrain_list[1] ** 0.5)
              + res.x[pe*2]*pred_nn_retrain_list[2] + res.x[pe*2+1]*(pred_nn_retrain_list[2] ** 2) + res.x[pe*2+2]*np.log(pred_nn_retrain_list[2]) + res.x[pe*2+3]*1/(1.0+pred_nn_retrain_list[2]) + res.x[pe*2+4]*(pred_nn_retrain_list[2] ** 0.5)
              + res.x[pe*3]*pred_nn_retrain_list[3] + res.x[pe*3+1]*(pred_nn_retrain_list[3] ** 2) + res.x[pe*3+2]*np.log(pred_nn_retrain_list[3]) + res.x[pe*3+3]*1/(1.0+pred_nn_retrain_list[3]) + res.x[pe*3+4]*(pred_nn_retrain_list[3] ** 0.5)
              + res.x[pe*4]*pred_nn_retrain_list[4] + res.x[pe*4+1]*(pred_nn_retrain_list[4] ** 2) + res.x[pe*4+2]*np.log(pred_nn_retrain_list[4]) + res.x[pe*4+3]*1/(1.0+pred_nn_retrain_list[4]) + res.x[pe*4+4]*(pred_nn_retrain_list[4] ** 0.5)
              + res.x[pe*5]*pred_nn_retrain_list[5] + res.x[pe*5+1]*(pred_nn_retrain_list[5] ** 2) + res.x[pe*5+2]*np.log(pred_nn_retrain_list[5]) + res.x[pe*5+3]*1/(1.0+pred_nn_retrain_list[5]) + res.x[pe*5+4]*(pred_nn_retrain_list[5] ** 0.5)
              + res.x[pe*6]*pred_nn_retrain_list[6] + res.x[pe*6+1]*(pred_nn_retrain_list[6] ** 2) + res.x[pe*6+2]*np.log(pred_nn_retrain_list[6]) + res.x[pe*6+3]*1/(1.0+pred_nn_retrain_list[6]) + res.x[pe*6+4]*(pred_nn_retrain_list[6] ** 0.5)
              + res.x[pe*7]*pred_nn_retrain_list[7] + res.x[pe*7+1]*(pred_nn_retrain_list[7] ** 2) + res.x[pe*7+2]*np.log(pred_nn_retrain_list[7]) + res.x[pe*7+3]*1/(1.0+pred_nn_retrain_list[7]) + res.x[pe*7+4]*(pred_nn_retrain_list[7] ** 0.5)
              + res.x[pe*8]*pred_nn_retrain_list[8] + res.x[pe*8+1]*(pred_nn_retrain_list[8] ** 2) + res.x[pe*8+2]*np.log(pred_nn_retrain_list[8]) + res.x[pe*8+3]*1/(1.0+pred_nn_retrain_list[8]) + res.x[pe*8+4]*(pred_nn_retrain_list[8] ** 0.5)
              + res.x[pe*9]*pred_nn_retrain_list[9] + res.x[pe*9+1]*(pred_nn_retrain_list[9] ** 2) + res.x[pe*9+2]*np.log(pred_nn_retrain_list[9]) + res.x[pe*9+3]*1/(1.0+pred_nn_retrain_list[9]) + res.x[pe*9+4]*(pred_nn_retrain_list[9] ** 0.5)
              + res.x[pe*10]*pred_nn_retrain_list[10] + res.x[pe*10+1]*(pred_nn_retrain_list[10] ** 2) + res.x[pe*10+2]*np.log(pred_nn_retrain_list[10]) + res.x[pe*10+3]*1/(1.0+pred_nn_retrain_list[10]) + res.x[pe*10+4]*(pred_nn_retrain_list[10] ** 0.5)
              + res.x[pe*11]*pred_nn_retrain_list[11] + res.x[pe*11+1]*(pred_nn_retrain_list[11] ** 2) + res.x[pe*11+2]*np.log(pred_nn_retrain_list[11]) + res.x[pe*11+3]*1/(1.0+pred_nn_retrain_list[11]) + res.x[pe*11+4]*(pred_nn_retrain_list[11] ** 0.5)
              + res.x[pe*12]*pred_nn_retrain_list[12] + res.x[pe*12+1]*(pred_nn_retrain_list[12] ** 2) + res.x[pe*12+2]*np.log(pred_nn_retrain_list[12]) + res.x[pe*12+3]*1/(1.0+pred_nn_retrain_list[12]) + res.x[pe*12+4]*(pred_nn_retrain_list[12] ** 0.5)
              + res.x[pe*13]*pred_nn_retrain_list[13] + res.x[pe*13+1]*(pred_nn_retrain_list[13] ** 2) + res.x[pe*13+2]*np.log(pred_nn_retrain_list[13]) + res.x[pe*13+3]*1/(1.0+pred_nn_retrain_list[13]) + res.x[pe*13+4]*(pred_nn_retrain_list[13] ** 0.5))


pred_nn = (res.x[pe*0]*pred_nn_test_list[0] + res.x[pe*0+1]*(pred_nn_test_list[0] ** 2) + res.x[pe*0+2]*np.log(pred_nn_test_list[0]) + res.x[pe*0+3]*1/(1.0+pred_nn_test_list[0]) + res.x[pe*0+4]*(pred_nn_test_list[0] ** 0.5)
              + res.x[pe*1]*pred_nn_test_list[1] + res.x[pe*1+1]*(pred_nn_test_list[1] ** 2) + res.x[pe*1+2]*np.log(pred_nn_test_list[1]) + res.x[pe*1+3]*1/(1.0+pred_nn_test_list[1]) + res.x[pe*1+4]*(pred_nn_test_list[1] ** 0.5)
              + res.x[pe*2]*pred_nn_test_list[2] + res.x[pe*2+1]*(pred_nn_test_list[2] ** 2) + res.x[pe*2+2]*np.log(pred_nn_test_list[2]) + res.x[pe*2+3]*1/(1.0+pred_nn_test_list[2]) + res.x[pe*2+4]*(pred_nn_test_list[2] ** 0.5)
              + res.x[pe*3]*pred_nn_test_list[3] + res.x[pe*3+1]*(pred_nn_test_list[3] ** 2) + res.x[pe*3+2]*np.log(pred_nn_test_list[3]) + res.x[pe*3+3]*1/(1.0+pred_nn_test_list[3]) + res.x[pe*3+4]*(pred_nn_test_list[3] ** 0.5)
              + res.x[pe*4]*pred_nn_test_list[4] + res.x[pe*4+1]*(pred_nn_test_list[4] ** 2) + res.x[pe*4+2]*np.log(pred_nn_test_list[4]) + res.x[pe*4+3]*1/(1.0+pred_nn_test_list[4]) + res.x[pe*4+4]*(pred_nn_test_list[4] ** 0.5)
              + res.x[pe*5]*pred_nn_test_list[5] + res.x[pe*5+1]*(pred_nn_test_list[5] ** 2) + res.x[pe*5+2]*np.log(pred_nn_test_list[5]) + res.x[pe*5+3]*1/(1.0+pred_nn_test_list[5]) + res.x[pe*5+4]*(pred_nn_test_list[5] ** 0.5)
              + res.x[pe*6]*pred_nn_test_list[6] + res.x[pe*6+1]*(pred_nn_test_list[6] ** 2) + res.x[pe*6+2]*np.log(pred_nn_test_list[6]) + res.x[pe*6+3]*1/(1.0+pred_nn_test_list[6]) + res.x[pe*6+4]*(pred_nn_test_list[6] ** 0.5)
              + res.x[pe*7]*pred_nn_test_list[7] + res.x[pe*7+1]*(pred_nn_test_list[7] ** 2) + res.x[pe*7+2]*np.log(pred_nn_test_list[7]) + res.x[pe*7+3]*1/(1.0+pred_nn_test_list[7]) + res.x[pe*7+4]*(pred_nn_test_list[7] ** 0.5)
              + res.x[pe*8]*pred_nn_test_list[8] + res.x[pe*8+1]*(pred_nn_test_list[8] ** 2) + res.x[pe*8+2]*np.log(pred_nn_test_list[8]) + res.x[pe*8+3]*1/(1.0+pred_nn_test_list[8]) + res.x[pe*8+4]*(pred_nn_test_list[8] ** 0.5)
              + res.x[pe*9]*pred_nn_test_list[9] + res.x[pe*9+1]*(pred_nn_test_list[9] ** 2) + res.x[pe*9+2]*np.log(pred_nn_test_list[9]) + res.x[pe*9+3]*1/(1.0+pred_nn_test_list[9]) + res.x[pe*9+4]*(pred_nn_test_list[9] ** 0.5)
              + res.x[pe*10]*pred_nn_test_list[10] + res.x[pe*10+1]*(pred_nn_test_list[10] ** 2) + res.x[pe*10+2]*np.log(pred_nn_test_list[10]) + res.x[pe*10+3]*1/(1.0+pred_nn_test_list[10]) + res.x[pe*10+4]*(pred_nn_test_list[10] ** 0.5)
              + res.x[pe*11]*pred_nn_test_list[11] + res.x[pe*11+1]*(pred_nn_test_list[11] ** 2) + res.x[pe*11+2]*np.log(pred_nn_test_list[11]) + res.x[pe*11+3]*1/(1.0+pred_nn_test_list[11]) + res.x[pe*11+4]*(pred_nn_test_list[11] ** 0.5)
              + res.x[pe*12]*pred_nn_test_list[12] + res.x[pe*12+1]*(pred_nn_test_list[12] ** 2) + res.x[pe*12+2]*np.log(pred_nn_test_list[12]) + res.x[pe*12+3]*1/(1.0+pred_nn_test_list[12]) + res.x[pe*12+4]*(pred_nn_test_list[12] ** 0.5)
              + res.x[pe*13]*pred_nn_test_list[13] + res.x[pe*13+1]*(pred_nn_test_list[13] ** 2) + res.x[pe*13+2]*np.log(pred_nn_test_list[13]) + res.x[pe*13+3]*1/(1.0+pred_nn_test_list[13]) + res.x[pe*13+4]*(pred_nn_test_list[13] ** 0.5))



##################




pe= 5
def f(coord,args):
    #pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,pred_11,pred_12,pred_13,pred_14,pred_15,pred_16,pred_17,pred_18,r = args
    return np.mean( np.abs(coord[pe*0]*args[0] + coord[pe*0+1]*(args[0] ** 2) + coord[pe*0+2]*np.log(args[0]) + coord[pe*0+3]*1/(1.0+args[0]) + coord[pe*0+4]*(args[0] ** 0.5)
                         + coord[pe*1]*args[1] + coord[pe*1+1]*(args[1] ** 2) + coord[pe*1+2]*np.log(args[1]) + coord[pe*1+3]*1/(1.0+args[1]) + coord[pe*1+4]*(args[1] ** 0.5)
                         + coord[pe*2]*args[2] + coord[pe*2+1]*(args[2] ** 2) + coord[pe*2+2]*np.log(args[2]) + coord[pe*2+3]*1/(1.0+args[2]) + coord[pe*2+4]*(args[2] ** 0.5) - args[-1]) )


initial_guess = np.array([0.1 for x in range(pe * 3)])


res = minimize(f,initial_guess,args = [
                                       pred_xgb_1_retrain['loss'].values, #2
                                       pred_xgb_2_retrain['loss'].values, #3
                                       pred_xgb_3_retrain['loss'].values, #4
                                       train['loss'].values
                                       ]
                              ,method='SLSQP', options={"maxiter":1000000,"disp":True})

print res

pred_xgb_test_list = [
                                       pred_xgb_1['loss'], #1
                                       pred_xgb_2['loss'], #2
                                       pred_xgb_3['loss'], #3
        ]

pred_xgb_retrain_list = [
                                       pred_xgb_1_retrain['loss'], #1
                                       pred_xgb_2_retrain['loss'], #2
                                       pred_xgb_3_retrain['loss'], #3
        ]

pred_xgb_retrain = (res.x[pe*0]*pred_xgb_retrain_list[0] + res.x[pe*0+1]*(pred_xgb_retrain_list[0] ** 2) + res.x[pe*0+2]*np.log(pred_xgb_retrain_list[0]) + res.x[pe*0+3]*1/(1.0+pred_xgb_retrain_list[0]) + res.x[pe*0+4]*(pred_xgb_retrain_list[0] ** 0.5)
              + res.x[pe*1]*pred_xgb_retrain_list[1] + res.x[pe*1+1]*(pred_xgb_retrain_list[1] ** 2) + res.x[pe*1+2]*np.log(pred_xgb_retrain_list[1]) + res.x[pe*1+3]*1/(1.0+pred_xgb_retrain_list[1]) + res.x[pe*1+4]*(pred_xgb_retrain_list[1] ** 0.5)
              + res.x[pe*2]*pred_xgb_retrain_list[2] + res.x[pe*2+1]*(pred_xgb_retrain_list[2] ** 2) + res.x[pe*2+2]*np.log(pred_xgb_retrain_list[2]) + res.x[pe*2+3]*1/(1.0+pred_xgb_retrain_list[2]) + res.x[pe*2+4]*(pred_xgb_retrain_list[2] ** 0.5))

pred_xgb = (res.x[pe*0]*pred_xgb_test_list[0] + res.x[pe*0+1]*(pred_xgb_test_list[0] ** 2) + res.x[pe*0+2]*np.log(pred_xgb_test_list[0]) + res.x[pe*0+3]*1/(1.0+pred_xgb_test_list[0]) + res.x[pe*0+4]*(pred_xgb_test_list[0] ** 0.5)
              + res.x[pe*1]*pred_xgb_test_list[1] + res.x[pe*1+1]*(pred_xgb_test_list[1] ** 2) + res.x[pe*1+2]*np.log(pred_xgb_test_list[1]) + res.x[pe*1+3]*1/(1.0+pred_xgb_test_list[1]) + res.x[pe*1+4]*(pred_xgb_test_list[1] ** 0.5)
              + res.x[pe*2]*pred_xgb_test_list[2] + res.x[pe*2+1]*(pred_xgb_test_list[2] ** 2) + res.x[pe*2+2]*np.log(pred_xgb_test_list[2]) + res.x[pe*2+3]*1/(1.0+pred_xgb_test_list[2]) + res.x[pe*2+4]*(pred_xgb_test_list[2] ** 0.5))






##############################

pe= 5
def f(coord,args):
    #pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,pred_11,pred_12,pred_13,pred_14,pred_15,pred_16,pred_17,pred_18,r = args
    return np.mean( np.abs(coord[pe*0]*args[0] + coord[pe*0+1]*(args[0] ** 2) + coord[pe*0+2]*np.log(args[0]) + coord[pe*0+3]*1/(1.0+args[0]) + coord[pe*0+4]*(args[0] ** 0.5)
                         + coord[pe*1]*args[1] + coord[pe*1+1]*(args[1] ** 2) + coord[pe*1+2]*np.log(args[1]) + coord[pe*1+3]*1/(1.0+args[1]) + coord[pe*1+4]*(args[1] ** 0.5)
                         + coord[pe*2]*(args[0] - args[1]) + coord[pe*2+1]*((args[0]-args[1]) ** 2) + coord[pe*2+2]*np.log(args[0]-args[1]) + coord[pe*2+3]*1/(1.0+args[0]-args[1]) + coord[pe*2+4]*(args[0]-args[1] ** 0.5)
                         + coord[pe*3]*(args[0] + args[1]) + coord[pe*3+1]*((args[0]+args[1]) ** 2) + coord[pe*3+2]*np.log(args[0]+args[1]) + coord[pe*3+3]*1/(1.0+args[0]+args[1]) + coord[pe*3+4]*(args[0]+args[1] ** 0.5)
                         + coord[pe*4]*(args[0]*args[1]) + coord[pe*4+1]*((args[0]*args[1]) ** 2) + coord[pe*4+2]*np.log(args[0]*args[1]) + coord[pe*4+3]*1/(1.0+args[0]*args[1]) + coord[pe*4+4]*(args[0]*args[1] ** 0.5)
                         + coord[pe*5]*(args[0]/args[1]) + coord[pe*5+1]*((args[0]/args[1]) ** 2) + coord[pe*5+2]*np.log(args[0]/args[1]) + coord[pe*5+3]*1/(1.0+args[0]/args[1]) + coord[pe*5+4]*(args[0]/args[1] ** 0.5)
                - args[-1]) )


initial_guess = np.array([0.1 for x in range(pe * 6)])


res = minimize(f,initial_guess,args = [
                                       pred_xgb_retrain.values, #2
                                       pred_nn_retrain.values, #3
                                       pred_xgb_retrain.values - pred_nn_retrain.values,
                                       pred_xgb_retrain.values + pred_nn_retrain.values,
                                       pred_xgb_retrain.values * pred_nn_retrain.values,
                                       pred_xgb_retrain.values / pred_nn_retrain.values,
                                       train['loss'].values
                                       ]
                              ,method='SLSQP', options={"maxiter":1000000,"disp":True})

print res




pred_ensemble_list = [
                                       pred_nn, #1
                                       pred_xgb_retrain
        ]

pred_ensemble = (res.x[pe*0]*pred_ensemble_list[0] + res.x[pe*0+1]*(pred_ensemble_list[0] ** 2) + res.x[pe*0+2]*np.log(pred_ensemble_list[0]) + res.x[pe*0+3]*1/(1.0+pred_ensemble_list[0]) + res.x[pe*0+4]*(pred_ensemble_list[0] ** 0.5)
              + res.x[pe*1]*pred_ensemble_list[1] + res.x[pe*1+1]*(pred_ensemble_list[1] ** 2) + res.x[pe*1+2]*np.log(pred_ensemble_list[1]) + res.x[pe*1+3]*1/(1.0+pred_ensemble_list[1]) + res.x[pe*1+4]*(pred_ensemble_list[1] ** 0.5)
              + res.x[pe*2]*pred_ensemble_list[2] + res.x[pe*2+1]*(pred_ensemble_list[2] ** 2) + res.x[pe*2+2]*np.log(pred_ensemble_list[2]) + res.x[pe*2+3]*1/(1.0+pred_ensemble_list[2]) + res.x[pe*2+4]*(pred_ensemble_list[2] ** 0.5)
              + res.x[pe*3]*pred_ensemble_list[3] + res.x[pe*3+1]*(pred_ensemble_list[3] ** 2) + res.x[pe*3+2]*np.log(pred_ensemble_list[3]) + res.x[pe*3+3]*1/(1.0+pred_ensemble_list[3]) + res.x[pe*3+4]*(pred_ensemble_list[3] ** 0.5)
              + res.x[pe*4]*pred_ensemble_list[4] + res.x[pe*4+1]*(pred_ensemble_list[4] ** 2) + res.x[pe*4+2]*np.log(pred_ensemble_list[4]) + res.x[pe*4+3]*1/(1.0+pred_ensemble_list[4]) + res.x[pe*4+4]*(pred_ensemble_list[4] ** 0.5)
              + res.x[pe*5]*pred_ensemble_list[5] + res.x[pe*5+1]*(pred_ensemble_list[5] ** 2) + res.x[pe*5+2]*np.log(pred_ensemble_list[5]) + res.x[pe*5+3]*1/(1.0+pred_ensemble_list[5]) + res.x[pe*5+4]*(pred_ensemble_list[5] ** 0.5))


pred_ensemble.to_csv("pred_retrain.csv", index_label='id')
