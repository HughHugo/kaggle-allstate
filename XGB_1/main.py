# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

ID = 'id'
TARGET = 'loss'
SEED = 6174
DATA_DIR = "../input"

TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)


train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

################################################################################
skf_table = pd.read_csv('../cache/stack_index.csv')
skf_table = pd.merge(train, skf_table, on='id')
skf = []
nfold = 10
for i in range(nfold):
    skf += [[(skf_table['stack_index']!=i).values, (skf_table['stack_index']==i).values]]
print skf
print len(skf)
label = np.log(train['loss'].values)
################################################################################


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

xgb_params = {
    'booster': 'gbtree',
    'seed': 6174,
    'silent': 1,
    'eta': 0.002,
    'objective': 'reg:linear',
    'max_depth': 10,
    'gamma': 2.89995505893,
    'min_child_weight': 1.03507430838,
    'max_delta_step': 1.84631913261,
    'subsample': 0.989412302457,
    'colsample_bytree': 0.305169996086,
    'colsample_bylevel': 0.969223656208,
    'lambda': 3.95933450744,
    'eval_metric': 'mae'
}

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

TRAIN_ROUNDS = 100000
gbdt = xgb.train(xgb_params, dtrain, TRAIN_ROUNDS)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = np.exp(gbdt.predict(dtest))
submission.to_csv('XGB_1.csv', index=None)

########################################################################################
sample = pd.read_csv(path+'sample_submission.csv')
submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
score = np.zeros(nfold)
i=0
for tr, te in skf:
    tr = np.where(tr)
    te = np.where(te)
    X_train, X_test, y_train, y_test = x_train[tr], x_train[te], label[tr], label[te]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    clf = xgb.train(xgb_params, dtrain, TRAIN_ROUNDS)
    dtest = xgb.DMatrix(X_test)
    pred = np.exp(clf.predict(dtest))
    tmp = pd.DataFrame(pred, columns=sample.columns[1:])
    submission.iloc[te[0],0] = pred
    score[i]= mean_absolute_error(np.exp(y_test), pred)
    print(score[i])
    i+=1

print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))
print(mean_absolute_error(np.exp(label),submission.values))

submission.to_csv("XGB_retrain_1.csv", index_label='id')
########################################################################################
