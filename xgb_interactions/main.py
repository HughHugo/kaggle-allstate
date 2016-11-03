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
trainID = train['id']
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
    'eta': 0.1,
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


gbdt = xgb.train(xgb_params, dtrain, 1000, obj=logregobj)

def create_feature_map(fmap_filename, features):
    """
    features: enumerable of feature names
    """
    outfile = open(fmap_filename, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

create_feature_map('./bin/xgb.fmap', features)
gbdt.dump_model('./bin/xgb.dump',fmap='./bin/xgb.fmap', with_stats=True)
