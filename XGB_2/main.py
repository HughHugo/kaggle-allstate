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

SHIFT = 200
################################################################################
skf_table = pd.read_csv('../cache/stack_index.csv')
skf_table = pd.merge(train, skf_table, on='id')
skf = []
nfold = 10
for i in range(nfold):
    skf += [[(skf_table['stack_index']!=i).values, (skf_table['stack_index']==i).values]]
print skf
print len(skf)
label = np.log(train['loss'].values + SHIFT)
################################################################################


y_train = np.log(train[TARGET].ravel() + SHIFT)

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

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con =2
    x =preds-labels
    grad =con*x / (np.abs(x)+con)
    hess =con**2 / (np.abs(x)+con)**2
    return grad, hess

xgb_params = {
    'min_child_weight': 1,
    'eta': 0.01,
    'colsample_bytree': 0.5,
    'max_depth': 12,
    'subsample': 0.8,
    'alpha': 1,
    'gamma': 1,
    'silent': 1,
    'verbose_eval': True,
    'seed': 6174,
}

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y) - SHIFT, np.exp(yhat) - SHIFT)

res = xgb.cv(xgb_params, dtrain, num_boost_round=999999999,
             nfold=4,
             seed=SEED,
             stratified=False, obj=logregobj,
             early_stopping_rounds=300,
             verbose_eval=10,
             show_stdv=True,
             feval=xg_eval_mae,
             maximize=False)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]
print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds, obj=logregobj)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = np.exp(gbdt.predict(dtest)) - SHIFT
submission.to_csv('XGB_2.csv', index=None)

########################################################################################
sample = pd.read_csv('../input/sample_submission.csv')
submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
score = np.zeros(nfold)
i=0
for tr, te in skf:
    tr = np.where(tr)
    te = np.where(te)
    X_train, X_test, y_train, y_test = x_train[tr], x_train[te], label[tr], label[te]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    clf = xgb.train(xgb_params, dtrain, best_nrounds, obj=logregobj)
    dtest = xgb.DMatrix(X_test)
    pred = np.exp(clf.predict(dtest)) - SHIFT
    tmp = pd.DataFrame(pred, columns=sample.columns[1:])
    submission.iloc[te[0],0] = pred
    score[i]= mean_absolute_error(np.exp(y_test), pred)
    print(score[i])
    i+=1

print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))
print(mean_absolute_error(np.exp(label) - SHIFT,submission.values))

submission.to_csv("XGB_retrain_2.csv", index_label='id')
########################################################################################
