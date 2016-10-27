# -*- coding: utf-8 -*-
"""
@author: Faron
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import os
import random

ID = 'id'
TARGET = 'loss'
SEED = 6174
DATA_DIR = "../input"

TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)


train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

y_train = np.log(train[TARGET].ravel())

train.drop([ID, TARGET], axis=1, inplace=True)
test.drop([ID], axis=1, inplace=True)

print("{},{}".format(train.shape, test.shape))

ntrain = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)

features = train.columns

cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]

print(train_test.head())

x_train = np.array(train_test.iloc[:ntrain,:])
x_test = np.array(train_test.iloc[ntrain:,:])

print("{},{}".format(train.shape, test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)


if os.path.isfile("../log/xgb_param.log"):
    print "1"
else:
    print "2"
    f=open("../log/xgb_param.log", "w")
    f.write("booster,eta,max_depth,gamma,min_child_weight,max_delta_step,subsample,colsample_bytree,colsample_bylevel,lambda,best_nrounds,cv_mean,cv_sd")
    f.write("\n")
    f.close()

while True:
    xgb_params = {
        'booster': 'gbtree',
        'seed': 6174,
        'silent': 1,
        'eta': 0.1,
        'objective': 'reg:linear',
        'max_depth': random.randint(3,20),
        'gamma': random.uniform(0, 5),
        'min_child_weight': random.uniform(0, 2),
        'max_delta_step': random.uniform(0, 5),
        'subsample': random.uniform(0.3, 1),
        'colsample_bytree': random.uniform(0.3, 1),
        'colsample_bylevel': random.uniform(0.3, 1),
        'lambda': random.uniform(0, 5),
        'eval_metric': 'mae'
    }

    def xg_eval_mae(yhat, dtrain):
        y = dtrain.get_label()
        return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

    res = xgb.cv(xgb_params, dtrain, num_boost_round=7500000, nfold=4, seed=SEED, stratified=False,
                 early_stopping_rounds=100, verbose_eval=10, show_stdv=True, feval=xg_eval_mae, maximize=False)

    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 0]
    cv_std = res.iloc[-1, 1]
    print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

    xgb_params['best_nrounds'] = best_nrounds
    xgb_params['cv_mean'] = cv_mean
    xgb_params['cv_std'] = cv_std
    f=open("../log/xgb_param.log", "a")
    f.write(str(xgb_params["booster"]) + ","+ str(xgb_params["eta"]) + "," +
         str(xgb_params["max_depth"]) + "," +
         str(xgb_params["gamma"]) + "," + str(xgb_params["min_child_weight"]) + "," +
         str(xgb_params["max_delta_step"]) + "," + str(xgb_params["subsample"]) + "," +
         str(xgb_params["colsample_bytree"]) + "," + str(xgb_params["colsample_bylevel"]) + "," +
         str(xgb_params["lambda"]) +  "," + str(xgb_params["best_nrounds"]) +  "," +
         str(xgb_params["cv_mean"]) +  "," + str(xgb_params["cv_std"]))
    f.write("\n")
    f.close()
