# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Main
pred_xgb_1_retrain = pd.read_csv('../XGB_2/XGB_retrain_2.csv', index_col=0)
pred_xgb_2_retrain = pd.read_csv('../XGB_3/XGB_retrain_3.csv', index_col=0)
pred_xgb_1 = pd.read_csv('../XGB_2/XGB_2.csv', index_col=0)
pred_xgb_2 = pd.read_csv('../XGB_3/XGB_3.csv', index_col=0)
train = pd.read_csv('../input/train.csv', index_col=0)

# optimize XGB
save_mae = 10000.0
save_i = None
for i in range(10001):
    tmp_loss = ((10000-i)*pred_xgb_1_retrain + i*pred_xgb_2_retrain)/10000.0
    tmp_mae = mean_absolute_error(train['loss'], tmp_loss)
    if tmp_mae < save_mae:
        #print tmp_mae
        save_mae = tmp_mae
        save_i = i

pred_xgb_retrain = ((10000-save_i)*pred_xgb_1_retrain + save_i*pred_xgb_2_retrain)/10000.0
pred_xgb = ((10000-save_i)*pred_xgb_1 + save_i*pred_xgb_2)/10000.0

print "\n"
print mean_absolute_error(train['loss'], pred_xgb_1_retrain)
print mean_absolute_error(train['loss'], pred_xgb_2_retrain)
print mean_absolute_error(train['loss'], pred_xgb_retrain)
print np.mean(pred_xgb_1_retrain)
print np.mean(pred_xgb_2_retrain)
print np.mean(pred_xgb_retrain)
print np.mean(pred_xgb_1)
print np.mean(pred_xgb_2)
print np.mean(pred_xgb)

pred_xgb_retrain.to_csv("./cache/pred_xgb_retrain.csv", index_label='id')
pred_xgb.to_csv("./cache/pred_xgb.csv", index_label='id')
