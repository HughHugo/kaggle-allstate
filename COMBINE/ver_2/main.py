# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Main
pred_nn_1_retrain = pd.read_csv('../../NN_1/NN_retrain_1.csv', index_col=0)
pred_nn_2_retrain = pd.read_csv('../../NN_2/NN_retrain_2.csv', index_col=0)
pred_nn_1 = pd.read_csv('../../NN_1/NN_1.csv', index_col=0)
pred_nn_2 = pd.read_csv('../../NN_2/NN_2.csv', index_col=0)
pred_xgb_1_retrain = pd.read_csv('../../XGB_2/XGB_retrain_2.csv', index_col=0)
pred_xgb_2_retrain = pd.read_csv('../../XGB_3/XGB_retrain_3.csv', index_col=0)
pred_xgb_1 = pd.read_csv('../../XGB_2/XGB_2.csv', index_col=0)
pred_xgb_2 = pd.read_csv('../../XGB_3/XGB_3.csv', index_col=0)
train = pd.read_csv('../../input/train.csv', index_col=0)


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
#1133.0687374
#1133.48770365
#1131.88010317

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

#1130.39688
#1128.13875418
#1128.1380479

# final
save_mae = 10000.0
save_i = None
for i in range(10001):
    tmp_loss = ((10000-i)*pred_nn_retrain + i*pred_xgb_retrain)/10000.0
    tmp_mae = mean_absolute_error(train['loss'], tmp_loss)
    if tmp_mae < save_mae:
        #print tmp_mae
        save_mae = tmp_mae
        save_i = i

pred = ((10000-save_i)*pred_nn + save_i*pred_xgb)/10000.0
print "\n"
print save_mae #1122.35798699
pred.to_csv("pred_retrain.csv", index_label='id')



tmp = ((10000-save_i)*pred_nn_retrain + save_i*pred_xgb_retrain)/10000.0
tmp.to_csv("retrain.csv", index_label='id')


# xgboost
#
# #2.2
# pred_1_retrain = pd.read_csv('../XGB_1/XGB_retrain_1.csv', index_col=0)
# pred_2_retrain = pd.read_csv('../XGB_2/XGB_retrain_2.csv', index_col=0)
#
# train = pd.read_csv('../input/train.csv', index_col=0)
#
# save_mae = 10000.0
# save_i = None
# for i in range(10001):
#     tmp_loss = ((10000-i)*pred_1_retrain + i*pred_2_retrain)/10000.0
#     tmp_mae = mean_absolute_error(train['loss'], tmp_loss)
#     if tmp_mae < save_mae:
#         print tmp_mae
#         save_mae = tmp_mae
#         save_i = i
#
# print np.mean(pred_1)
# print np.mean(pred_2)
#
# pred = ((10000-save_i)*pred_1 + save_i*pred_2)/10000.0
#
#
#
#
#
#
# #3
# pred_1_retrain = pd.read_csv('../NN_1/NN_retrain_1.csv', index_col=0)
# pred_2_retrain = pd.read_csv('../XGB_2/XGB_retrain_2.csv', index_col=0)
# pred_3_retrain = pd.read_csv('../XGB_1/XGB_retrain_1.csv', index_col=0)
#
# print np.mean(pred_1_retrain)
# print np.mean(pred_2_retrain)
# print np.mean(pred_3_retrain)
#
# train = pd.read_csv('../input/train.csv', index_col=0)
#
# save_mae = 10000.0
# save_i = None
# save_j = None
# save_k = None
#
# for i in range(1001):
#     for j in range(1001-i):
#         k = 1000-i-j
#         tmp_loss = (i*pred_1_retrain + j*pred_2_retrain +k*pred_3_retrain)/1000.0
#         tmp_mae = mean_absolute_error(train['loss'], tmp_loss)
#         if tmp_mae < save_mae:
#             print tmp_mae
#             #print i+j+k
#             save_mae = tmp_mae
#             save_i = i
#             save_j = j
#             save_k = k
#
# print np.mean(pred_1)
# print np.mean(pred_2)
#
# #pred = ((10000-save_i-save_j)*pred_1 + save_i*pred_2 + save_j*pred_3_retrain)/10000.0
# #1123.63547286
# #pred.to_csv("pred_retrain.csv", index_label='id')
