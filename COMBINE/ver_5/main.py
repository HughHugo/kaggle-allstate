# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Main
pred_nn_1_retrain = pd.read_csv('../../NN_1_fix/NN_retrain_1.csv', index_col=0)
pred_nn_2_retrain = pd.read_csv('../../NN_2_fix/NN_retrain_2.csv', index_col=0)
pred_nn_3_retrain = pd.read_csv('../../NN_3_fix/NN_retrain_3.csv', index_col=0)

pred_nn_1 = pd.read_csv('../../NN_1_fix/NN_1.csv', index_col=0)
pred_nn_2 = pd.read_csv('../../NN_2_fix/NN_2.csv', index_col=0)
pred_nn_3 = pd.read_csv('../../NN_3_fix/NN_3.csv', index_col=0)

pred_xgb_1_retrain = pd.read_csv('../../XGB_1/XGB_retrain_1.csv', index_col=0)
pred_xgb_2_retrain = pd.read_csv('../../XGB_2/XGB_retrain_2.csv', index_col=0)
pred_xgb_3_retrain = pd.read_csv('../../XGB_3/XGB_retrain_3.csv', index_col=0)

pred_xgb_1 = pd.read_csv('../../XGB_1/XGB_1.csv', index_col=0)
pred_xgb_2 = pd.read_csv('../../XGB_2/XGB_2.csv', index_col=0)
pred_xgb_3 = pd.read_csv('../../XGB_3/XGB_3.csv', index_col=0)

train = pd.read_csv('../../input/train.csv', index_col=0)

# Check MAE Local
print mean_absolute_error(train['loss'], pred_nn_1_retrain)
print mean_absolute_error(train['loss'], pred_nn_2_retrain)
print mean_absolute_error(train['loss'], pred_nn_3_retrain)
print mean_absolute_error(train['loss'], pred_xgb_1_retrain)
print mean_absolute_error(train['loss'], pred_xgb_2_retrain)
print mean_absolute_error(train['loss'], pred_xgb_3_retrain)
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
    pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,r = args
    return np.mean( np.abs(coord[0]*pred_1 + coord[1]*pred_2 + coord[2]*pred_3
                          +coord[3]*pred_4 + coord[4]*pred_5 + coord[5]*pred_6  - r))


initial_guess = np.array([0.2,0.2,0.2,0.2,0.2,0.2])


res = minimize(f,initial_guess,args = [pred_nn_1_retrain['loss'].values,
                                       pred_nn_2_retrain['loss'].values,
                                       pred_nn_3_retrain['loss'].values,
                                       pred_xgb_1_retrain['loss'].values,
                                       pred_xgb_2_retrain['loss'].values,
                                       pred_xgb_3_retrain['loss'].values,
                                       train['loss'].values]
                              ,method='SLSQP')


print res
pred_ensemble = (res.x[0]*pred_nn_1 + res.x[1]*pred_nn_2 + res.x[2]*pred_nn_3
              + res.x[3]*pred_xgb_1 + res.x[4]*pred_xgb_2 + res.x[5]*pred_xgb_3)


pred_ensemble.to_csv("pred_retrain.csv", index_label='id')