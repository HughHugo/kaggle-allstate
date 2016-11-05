# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Main
pred_nn_1_retrain = pd.read_csv('../NN_1_fix/NN_retrain_1.csv', index_col=0)
pred_nn_2_retrain = pd.read_csv('../NN_2_fix/NN_retrain_2.csv', index_col=0)
#pred_nn_3_retrain = pd.read_csv('../NN_3_fix/NN_retrain_3.csv', index_col=0)
pred_nn_1 = pd.read_csv('../NN_1_fix/NN_1.csv', index_col=0)
pred_nn_2 = pd.read_csv('../NN_2_fix/NN_2.csv', index_col=0)
train = pd.read_csv('../input/train.csv', index_col=0)

# optimize NN
save_mae = 10000.0
save_i = None
for i in range(10001):
    tmp_loss = ((10000-i)*pred_nn_1_retrain + i*pred_nn_2_retrain)/10000.0
    tmp_mae = mean_absolute_error(train['loss'], tmp_loss)
    if tmp_mae < save_mae:
        #print tmp_mae
        save_mae = tmp_mae
        save_i = i


pred_nn_retrain = ((10000-save_i)*pred_nn_1_retrain + save_i*pred_nn_2_retrain)/10000.0
pred_nn = ((10000-save_i)*pred_nn_1 + save_i*pred_nn_2)/10000.0
print "\n"
print mean_absolute_error(train['loss'], pred_nn_1_retrain)
print mean_absolute_error(train['loss'], pred_nn_2_retrain)
print mean_absolute_error(train['loss'], pred_nn_retrain)
print np.mean(pred_nn_1_retrain)
print np.mean(pred_nn_2_retrain)
print np.mean(pred_nn_retrain)
print np.mean(pred_nn_1)
print np.mean(pred_nn_2)
print np.mean(pred_nn)

pred_nn_retrain.to_csv("./cache/pred_nn_retrain.csv", index_label='id')
pred_nn.to_csv("./cache/pred_nn.csv", index_label='id')
