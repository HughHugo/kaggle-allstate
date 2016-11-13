# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('../../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
# Main
pred_combine_1_retrain = pd.read_csv('../../COMBINE/ver_1/retrain.csv', index_col=0)
pred_combine_2_retrain = pd.read_csv('../../COMBINE/ver_2/retrain.csv', index_col=0)
pred_combine_3_retrain = pd.read_csv('../../COMBINE/ver_3/retrain.csv', index_col=0)
pred_combine_4_retrain = pd.read_csv('../../COMBINE/ver_4/retrain.csv', index_col=0)
pred_combine_5_retrain = pd.read_csv('../../COMBINE/ver_5/retrain.csv', index_col=0)
pred_combine_6_retrain = pd.read_csv('../../COMBINE/ver_6/retrain.csv', index_col=0)
pred_combine_7_retrain = pd.read_csv('../../COMBINE/ver_7/retrain.csv', index_col=0)
pred_combine_8_retrain = pd.read_csv('../../COMBINE/ver_8/retrain.csv', index_col=0)
pred_combine_9_retrain = pd.read_csv('../../COMBINE/ver_9/retrain.csv', index_col=0)
pred_combine_10_retrain = pd.read_csv('../../COMBINE/ver_10/retrain.csv', index_col=0)
pred_combine_11_retrain = pd.read_csv('../../COMBINE/ver_11/retrain.csv', index_col=0)
pred_combine_12_retrain = pd.read_csv('../../COMBINE/ver_12/retrain.csv', index_col=0)
pred_combine_13_retrain = pd.read_csv('../../COMBINE/ver_13/retrain.csv', index_col=0)
pred_combine_14_retrain = pd.read_csv('../../COMBINE/ver_14/retrain.csv', index_col=0)
pred_combine_15_retrain = pd.read_csv('../../COMBINE/ver_15/retrain.csv', index_col=0)
pred_combine_21_retrain = pd.read_csv('../../COMBINE/ver_21/retrain.csv', index_col=0)
pred_combine_22_retrain = pd.read_csv('../../COMBINE/ver_22/retrain.csv', index_col=0)


pred_combine_1 = pd.read_csv('../../COMBINE/ver_1/pred_retrain.csv', index_col=0)
pred_combine_2 = pd.read_csv('../../COMBINE/ver_2/pred_retrain.csv', index_col=0)
pred_combine_3 = pd.read_csv('../../COMBINE/ver_3/pred_retrain.csv', index_col=0)
pred_combine_4 = pd.read_csv('../../COMBINE/ver_4/pred_retrain.csv', index_col=0)
pred_combine_5 = pd.read_csv('../../COMBINE/ver_5/pred_retrain.csv', index_col=0)
pred_combine_6 = pd.read_csv('../../COMBINE/ver_6/pred_retrain.csv', index_col=0)
pred_combine_7 = pd.read_csv('../../COMBINE/ver_7/pred_retrain.csv', index_col=0)
pred_combine_8 = pd.read_csv('../../COMBINE/ver_8/pred_retrain.csv', index_col=0)
pred_combine_9 = pd.read_csv('../../COMBINE/ver_9/pred_retrain.csv', index_col=0)
pred_combine_10 = pd.read_csv('../../COMBINE/ver_10/pred_retrain.csv', index_col=0)
pred_combine_11 = pd.read_csv('../../COMBINE/ver_11/pred_retrain.csv', index_col=0)
pred_combine_12 = pd.read_csv('../../COMBINE/ver_12/pred_retrain.csv', index_col=0)
pred_combine_13 = pd.read_csv('../../COMBINE/ver_13/pred_retrain.csv', index_col=0)
pred_combine_14 = pd.read_csv('../../COMBINE/ver_14/pred_retrain.csv', index_col=0)
pred_combine_15 = pd.read_csv('../../COMBINE/ver_15/pred_retrain.csv', index_col=0)
pred_combine_21 = pd.read_csv('../../COMBINE/ver_21/pred_retrain.csv', index_col=0)
pred_combine_22 = pd.read_csv('../../COMBINE/ver_22/pred_retrain.csv', index_col=0)

print mean_absolute_error(train['loss'], pred_combine_1_retrain)
print mean_absolute_error(train['loss'], pred_combine_2_retrain)
print mean_absolute_error(train['loss'], pred_combine_3_retrain)
print mean_absolute_error(train['loss'], pred_combine_4_retrain)
print mean_absolute_error(train['loss'], pred_combine_5_retrain)
print mean_absolute_error(train['loss'], pred_combine_6_retrain)
print mean_absolute_error(train['loss'], pred_combine_7_retrain)
print mean_absolute_error(train['loss'], pred_combine_8_retrain)
print mean_absolute_error(train['loss'], pred_combine_9_retrain)
print mean_absolute_error(train['loss'], pred_combine_10_retrain)
print mean_absolute_error(train['loss'], pred_combine_11_retrain)
print mean_absolute_error(train['loss'], pred_combine_12_retrain)
print mean_absolute_error(train['loss'], pred_combine_13_retrain)
print mean_absolute_error(train['loss'], pred_combine_14_retrain)
print mean_absolute_error(train['loss'], pred_combine_15_retrain)
print mean_absolute_error(train['loss'], pred_combine_21_retrain)
print mean_absolute_error(train['loss'], pred_combine_22_retrain)

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
                         + coord[pe*14]*args[14] + coord[pe*14+1]*(args[14] ** 2) + coord[pe*14+2]*np.log(args[14]) + coord[pe*14+3]*1/(1.0+args[14]) + coord[pe*14+4]*(args[14] ** 0.5)
                         + coord[pe*15]*args[15] + coord[pe*15+1]*(args[15] ** 2) + coord[pe*15+2]*np.log(args[15]) + coord[pe*15+3]*1/(1.0+args[15]) + coord[pe*15+4]*(args[15] ** 0.5)
                         + coord[pe*16]*args[16] + coord[pe*16+1]*(args[16] ** 2) + coord[pe*16+2]*np.log(args[16]) + coord[pe*16+3]*1/(1.0+args[16]) + coord[pe*16+4]*(args[16] ** 0.5)
                         + coord[pe*17]
                         - args[-1]) )


initial_guess = np.array([0.1 for x in range(pe * 17 + 1)])


res = minimize(f,initial_guess,args = [
                                       pred_combine_1_retrain['loss'].values,    #1
                                       pred_combine_2_retrain['loss'].values,    #2
                                       pred_combine_3_retrain['loss'].values,    #3
                                       pred_combine_4_retrain['loss'].values,    #4
                                       pred_combine_5_retrain['loss'].values,    #5
                                       pred_combine_6_retrain['loss'].values,    #6
                                       pred_combine_7_retrain['loss'].values,    #7
                                       pred_combine_8_retrain['loss'].values,    #8
                                       pred_combine_9_retrain['loss'].values,    #9
                                       pred_combine_10_retrain['loss'].values,    #10
                                       pred_combine_11_retrain['loss'].values,    #11
                                       pred_combine_12_retrain['loss'].values,    #12
                                       pred_combine_13_retrain['loss'].values,    #13
                                       pred_combine_14_retrain['loss'].values,    #14
                                       pred_combine_15_retrain['loss'].values,    #15
                                       pred_combine_21_retrain['loss'].values,    #16
                                       pred_combine_22_retrain['loss'].values,    #17
                                       train['loss'].values
                                       ]
                              ,method='SLSQP', options={"maxiter":1000000,"disp":True})

print res

####### prediction

args = [
                                       pred_combine_1,
                                       pred_combine_2,
                                       pred_combine_3,
                                       pred_combine_4,
                                       pred_combine_5,
                                       pred_combine_6,
                                       pred_combine_7,
                                       pred_combine_8,
                                       pred_combine_9,
                                       pred_combine_10,
                                       pred_combine_11,
                                       pred_combine_12,
                                       pred_combine_13,
                                       pred_combine_14,
                                       pred_combine_15,
                                       pred_combine_21,
                                       pred_combine_22,

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
                     + res.x[pe*17]
            )

pred_ensemble = pd.DataFrame(pred_ensemble)
pred_ensemble.to_csv("pred_retrain.csv", index_label='id')
#
#
# args = [
#                                        pred_nn_1_fix_retrain['loss'],    #1
#                                        pred_nn_2_fix_retrain['loss'],    #2
#                                        pred_nn_3_fix_retrain['loss'],    #3
#                                        pred_nn_4_fix_retrain['loss'],    #4
#                                        pred_nn_5_fix_retrain['loss'],    #5
#                                        pred_nn_6_fix_retrain['loss'],    #6
#                                        pred_new_nn_1_retrain['loss'],    #7
#                                        pred_new_nn_2_retrain['loss'],    #8
#                                        pred_new_nn_3_retrain['loss'],    #9
#                                        pred_new_nn_4_retrain['loss'],    #10
#                                        pred_new_nn_5_retrain['loss'],    #11
#                                        pred_new_nn_1_65_retrain['loss'], #12
#                                        pred_new_nn_2_65_retrain['loss'], #13
#                                        pred_xgb_1_retrain['loss'],       #14
#                                        pred_xgb_2_retrain['loss'],       #15
#                                        pred_xgb_3_retrain['loss'],       #16
#                                        pred_nn_1_retrain['loss'],        #17
#                                        pred_nn_2_retrain['loss'],        #18
#                                        pred_nn_3_retrain['loss'],        #19
#                                        pred_nn_4_retrain['loss'],        #20
#                                        pred_nn_5_retrain['loss'],        #21
#                                        pred_nn_6_retrain['loss'],        #22
#                                        train['loss']
#                                        ]
#
# tmp = (res.x[pe*0]*args[0] + res.x[pe*0+1]*(args[0] ** 2) + res.x[pe*0+2]*np.log(args[0]) + res.x[pe*0+3]*1/(1.0+args[0]) + res.x[pe*0+4]*(args[0] ** 0.5)
#               + res.x[pe*1]*args[1] + res.x[pe*1+1]*(args[1] ** 2) + res.x[pe*1+2]*np.log(args[1]) + res.x[pe*1+3]*1/(1.0+args[1]) + res.x[pe*1+4]*(args[1] ** 0.5)
#               + res.x[pe*2]*args[2] + res.x[pe*2+1]*(args[2] ** 2) + res.x[pe*2+2]*np.log(args[2]) + res.x[pe*2+3]*1/(1.0+args[2]) + res.x[pe*2+4]*(args[2] ** 0.5)
#               + res.x[pe*3]*args[3] + res.x[pe*3+1]*(args[3] ** 2) + res.x[pe*3+2]*np.log(args[3]) + res.x[pe*3+3]*1/(1.0+args[3]) + res.x[pe*3+4]*(args[3] ** 0.5)
#               + res.x[pe*4]*args[4] + res.x[pe*4+1]*(args[4] ** 2) + res.x[pe*4+2]*np.log(args[4]) + res.x[pe*4+3]*1/(1.0+args[4]) + res.x[pe*4+4]*(args[4] ** 0.5)
#               + res.x[pe*5]*args[5] + res.x[pe*5+1]*(args[5] ** 2) + res.x[pe*5+2]*np.log(args[5]) + res.x[pe*5+3]*1/(1.0+args[5]) + res.x[pe*5+4]*(args[5] ** 0.5)
#               + res.x[pe*6]*args[6] + res.x[pe*6+1]*(args[6] ** 2) + res.x[pe*6+2]*np.log(args[6]) + res.x[pe*6+3]*1/(1.0+args[6]) + res.x[pe*6+4]*(args[6] ** 0.5)
#               + res.x[pe*7]*args[7] + res.x[pe*7+1]*(args[7] ** 2) + res.x[pe*7+2]*np.log(args[7]) + res.x[pe*7+3]*1/(1.0+args[7]) + res.x[pe*7+4]*(args[7] ** 0.5)
#               + res.x[pe*8]*args[8] + res.x[pe*8+1]*(args[8] ** 2) + res.x[pe*8+2]*np.log(args[8]) + res.x[pe*8+3]*1/(1.0+args[8]) + res.x[pe*8+4]*(args[8] ** 0.5)
#               + res.x[pe*9]*args[9] + res.x[pe*9+1]*(args[9] ** 2) + res.x[pe*9+2]*np.log(args[9]) + res.x[pe*9+3]*1/(1.0+args[9]) + res.x[pe*9+4]*(args[9] ** 0.5)
#               + res.x[pe*10]*args[10] + res.x[pe*10+1]*(args[10] ** 2) + res.x[pe*10+2]*np.log(args[10]) + res.x[pe*10+3]*1/(1.0+args[10]) + res.x[pe*10+4]*(args[10] ** 0.5)
#               + res.x[pe*11]*args[11] + res.x[pe*11+1]*(args[11] ** 2) + res.x[pe*11+2]*np.log(args[11]) + res.x[pe*11+3]*1/(1.0+args[11]) + res.x[pe*11+4]*(args[11] ** 0.5)
#               + res.x[pe*12]*args[12] + res.x[pe*12+1]*(args[12] ** 2) + res.x[pe*12+2]*np.log(args[12]) + res.x[pe*12+3]*1/(1.0+args[12]) + res.x[pe*12+4]*(args[12] ** 0.5)
#               + res.x[pe*13]*args[13] + res.x[pe*13+1]*(args[13] ** 2) + res.x[pe*13+2]*np.log(args[13]) + res.x[pe*13+3]*1/(1.0+args[13]) + res.x[pe*13+4]*(args[13] ** 0.5)
#               + res.x[pe*14]*args[14] + res.x[pe*14+1]*(args[14] ** 2) + res.x[pe*14+2]*np.log(args[14]) + res.x[pe*14+3]*1/(1.0+args[14]) + res.x[pe*14+4]*(args[14] ** 0.5)
#               + res.x[pe*15]*args[15] + res.x[pe*15+1]*(args[15] ** 2) + res.x[pe*15+2]*np.log(args[15]) + res.x[pe*15+3]*1/(1.0+args[15]) + res.x[pe*15+4]*(args[15] ** 0.5)
#               + res.x[pe*16]*args[16] + res.x[pe*16+1]*(args[16] ** 2) + res.x[pe*16+2]*np.log(args[16]) + res.x[pe*16+3]*1/(1.0+args[16]) + res.x[pe*16+4]*(args[16] ** 0.5)
#               + res.x[pe*17]*args[17] + res.x[pe*17+1]*(args[17] ** 2) + res.x[pe*17+2]*np.log(args[17]) + res.x[pe*17+3]*1/(1.0+args[17]) + res.x[pe*17+4]*(args[17] ** 0.5)
#               + res.x[pe*18]*args[18] + res.x[pe*18+1]*(args[18] ** 2) + res.x[pe*18+2]*np.log(args[18]) + res.x[pe*18+3]*1/(1.0+args[18]) + res.x[pe*18+4]*(args[18] ** 0.5)
#               + res.x[pe*19]*args[19] + res.x[pe*19+1]*(args[19] ** 2) + res.x[pe*19+2]*np.log(args[19]) + res.x[pe*19+3]*1/(1.0+args[19]) + res.x[pe*19+4]*(args[19] ** 0.5)
#            + res.x[pe*20]*args[20] + res.x[pe*20+1]*(args[20] ** 2) + res.x[pe*20+2]*np.log(args[20]) + res.x[pe*20+3]*1/(1.0+args[20]) + res.x[pe*20+4]*(args[20] ** 0.5)
#            + res.x[pe*21]*args[21] + res.x[pe*21+1]*(args[21] ** 2) + res.x[pe*21+2]*np.log(args[21]) + res.x[pe*21+3]*1/(1.0+args[21]) + res.x[pe*21+4]*(args[21] ** 0.5)
#            + res.x[pe*22]*(args[15] - args[10])
#            + res.x[pe*22+1]*((args[15]-args[10]) ** 2)
#            + res.x[pe*22+2]*np.log(abs(args[15]-args[10]))
#            + res.x[pe*22+3]*1/(1.0+abs(args[15]-args[10]))
#            + res.x[pe*22+4]*(abs(args[15]-args[10]) ** 0.5)
#            + res.x[pe*22+5]
#             )
# tmp = pd.DataFrame(tmp)
# tmp.to_csv("retrain.csv")
