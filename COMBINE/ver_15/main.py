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

pred_xgb_1_retrain = pd.read_csv('../../XGB_1/XGB_retrain_1.csv', index_col=0)
pred_xgb_2_retrain = pd.read_csv('../../XGB_2/XGB_retrain_2.csv', index_col=0)
pred_xgb_3_retrain = pd.read_csv('../../XGB_3/XGB_retrain_3.csv', index_col=0)

pred_xgb_1 = pd.read_csv('../../XGB_1/XGB_1.csv', index_col=0)
pred_xgb_2 = pd.read_csv('../../XGB_2/XGB_2.csv', index_col=0)
pred_xgb_3 = pd.read_csv('../../XGB_3/XGB_3.csv', index_col=0)

train = pd.read_csv('../../input/train.csv', index_col=0)

# Check MAE Local
#print mean_absolute_error(train['loss'], pred_nn_1_retrain)
#print mean_absolute_error(train['loss'], pred_nn_2_retrain)
#print mean_absolute_error(train['loss'], pred_nn_3_retrain)
#print mean_absolute_error(train['loss'], pred_nn_4_retrain)
#print mean_absolute_error(train['loss'], pred_nn_5_retrain)
#print mean_absolute_error(train['loss'], pred_nn_6_retrain)

print mean_absolute_error(train['loss'], pred_nn_1_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_2_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_3_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_4_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_5_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_6_fix_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_1_retrain)

#1
# save_mae = 10000.0
# save_i = None
# for i in range(10001):
#     tmp_loss = ((10000-i)*pred_nn_1_retrain + i*pred_xgb_3_retrain)/10000.0
#     tmp_mae = mean_absolute_error(train['loss'], tmp_loss)
#     if tmp_mae < save_mae:
#         print tmp_mae
#         save_mae = tmp_mae
#         save_i = i
#
#
# #2
# pred_retrain = pd.concat([
#                             pred_nn_1_retrain,
#                             pred_xgb_1_retrain,
#                             pred_xgb_2_retrain,
#                             pred_xgb_3_retrain,
#                         ], axis=1)
#
# pred_retrain.columns =  ['f_' + str(i) for i in range(len(pred_retrain.columns))]
#
#
# SHIFT = 200
# SEED = 6174
# pred_retrain_y = np.log(train['loss'].values + SHIFT)
#
#
# dtrain = xgb.DMatrix(pred_retrain, label=pred_retrain_y)
#
# def logregobj(preds, dtrain):
#     labels = dtrain.get_label()
#     con =2
#     x =preds-labels
#     grad =con*x / (np.abs(x)+con)
#     hess =con**2 / (np.abs(x)+con)**2
#     return grad, hess
#
# xgb_params = {
#     'min_child_weight': 1,
#     'eta': 0.1,
#     'colsample_bytree': 1,
#     'max_depth': 3,
#     'subsample': 0.7,
#     'alpha': 0,
#     'gamma': 0,
#     'silent': 1,
#     'verbose_eval': True,
#     'seed': 6174,
# }
#
# def xg_eval_mae(yhat, dtrain):
#     y = dtrain.get_label()
#     return 'mae', mean_absolute_error(np.exp(y) - SHIFT, np.exp(yhat) - SHIFT)
#
# res = xgb.cv(xgb_params, dtrain, num_boost_round=9999999,
#              nfold=10,
#              seed=SEED,
#              stratified=False, obj=logregobj,
#              early_stopping_rounds=300,
#              verbose_eval=10,
#              show_stdv=True,
#              feval=xg_eval_mae,
#              maximize=False)

# 3
from scipy.optimize import minimize

# ======================== NN optimize ======================== #
def f(coord,args):
    pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,pred_11,pred_12,pred_13,pred_14,pred_15,pred_16,pred_17,pred_18,r = args
    return np.mean( np.abs(coord[0]*pred_1 + coord[1]*pred_2 + coord[2]*pred_3
                          +coord[3]*pred_4 + coord[4]*pred_5 + coord[5]*pred_6
                          +coord[6]*pred_7 + coord[7]*pred_8 + coord[8]*pred_9
                          +coord[9]*pred_10
                          +coord[10]*pred_11 + coord[11]*pred_12 + coord[12]*pred_13
                          +coord[13]*pred_14 + coord[14]*pred_15 + coord[15]*pred_16
                          +coord[16]*(pred_17 ** 2) + coord[17]*np.log(pred_18)
                          - r))


initial_guess = np.array([0.1 for x in range(18)])


res = minimize(f,initial_guess,args = [
                                       pred_nn_1_fix_retrain['loss'].values,
                                       pred_nn_2_fix_retrain['loss'].values,
                                       pred_nn_3_fix_retrain['loss'].values,
                                       pred_nn_4_fix_retrain['loss'].values,
                                       pred_nn_5_fix_retrain['loss'].values,
                                       pred_nn_6_fix_retrain['loss'].values,
                                       pred_new_nn_1_retrain['loss'].values,
                                       pred_xgb_1_retrain['loss'].values,
                                       pred_xgb_2_retrain['loss'].values,
                                       pred_xgb_3_retrain['loss'].values,
                                       pred_nn_1_retrain['loss'].values,
                                       pred_nn_2_retrain['loss'].values,
                                       pred_nn_3_retrain['loss'].values,
                                       pred_nn_4_retrain['loss'].values,
                                       pred_nn_5_retrain['loss'].values,
                                       pred_nn_6_retrain['loss'].values,
                                       pred_xgb_3_retrain['loss'].values,
                                       pred_xgb_3_retrain['loss'].values,
                                       train['loss'].values]
                              ,method='SLSQP')

print res
pred_ensemble = (res.x[0]*pred_nn_1_fix + res.x[1]*pred_nn_2_fix + res.x[2]*pred_nn_3_fix
               + res.x[3]*pred_nn_4_fix + res.x[4]*pred_nn_5_fix + res.x[5]*pred_nn_6_fix
               + res.x[6]*pred_new_nn_1 + res.x[7]*pred_xgb_1 + res.x[8]*pred_xgb_2
               + res.x[9]*pred_xgb_3
               + res.x[10]*pred_nn_1 + res.x[11]*pred_nn_2 + res.x[12]*pred_nn_3
               + res.x[13]*pred_nn_4 + res.x[14]*pred_nn_5 + res.x[15]*pred_nn_6
               + res.x[16]*(pred_xgb_3 ** 2) + res.x[17]*np.log(pred_xgb_3)
               )


pred_ensemble.to_csv("pred_retrain.csv", index_label='id')


tmp = (res.x[0]*pred_nn_1_fix_retrain + res.x[1]*pred_nn_2_fix_retrain + res.x[2]*pred_nn_3_fix_retrain
               + res.x[3]*pred_nn_4_fix_retrain + res.x[4]*pred_nn_5_fix_retrain + res.x[5]*pred_nn_6_fix_retrain
               + res.x[6]*pred_new_nn_1_retrain + res.x[7]*pred_xgb_1_retrain + res.x[8]*pred_xgb_2_retrain
               + res.x[9]*pred_xgb_3_retrain
               + res.x[10]*pred_nn_1_retrain + res.x[11]*pred_nn_2_retrain + res.x[12]*pred_nn_3_retrain
               + res.x[13]*pred_nn_4_retrain + res.x[14]*pred_nn_5_retrain + res.x[15]*pred_nn_6_retrain
               + res.x[16]*(pred_xgb_3_retrain ** 2) + res.x[17]*np.log(pred_xgb_3_retrain)
               )


tmp.to_csv("retrain.csv", index_label='id')
