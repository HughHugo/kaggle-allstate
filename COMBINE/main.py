# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

#1
pred_1 = pd.read_csv('../NN_1/NN_1.csv', index_col=0)
pred_2 = pd.read_csv('../XGB_2/XGB_2.csv', index_col=0)

print np.mean(pred_1)
print np.mean(pred_2)

pred = (pred_1 + pred_2)/2.0
pred.to_csv("pred.csv", index_label='id')

#2
pred_1_retrain = pd.read_csv('../NN_1/NN_retrain_1.csv', index_col=0)
pred_2_retrain = pd.read_csv('../XGB_2/XGB_retrain_2.csv', index_col=0)

train = pd.read_csv('../input/train.csv', index_col=0)

save_mae = 10000.0
save_i = None
for i in range(10001):
    tmp_loss = ((10000-i)*pred_1_retrain + i*pred_2_retrain)/10000.0
    tmp_mae = mean_absolute_error(train['loss'], tmp_loss)
    if tmp_mae < save_mae:
        print tmp_mae
        save_mae = tmp_mae
        save_i = i

print np.mean(pred_1)
print np.mean(pred_2)

pred = ((10000-save_i)*pred_1 + save_i*pred_2)/10000.0
#1123.63547286
pred.to_csv("pred_retrain.csv", index_label='id')

#3
