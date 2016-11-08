"""
Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization
import pickle
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold

ID = 'id'
TARGET = 'loss'
SEED = 6174
DATA_DIR = "../input"

TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)

SHIFT = 200
##########################################################################################
## read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

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

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
y_train = np.log(train['loss'].values + SHIFT)
id_train = train['id'].values
id_test = test['id'].values

trainID = id_train


## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

######## Add continuous #########
for f in f_cat:
    tr_te[f] = pd.factorize(tr_te[f], sort=True)[0]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_cat]))
#################################

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

#x_train = np.array(train_test.iloc[:ntrain,:])
#x_test = np.array(train_test.iloc[ntrain:,:])
#print("{},{}".format(train.shape, test.shape))

dtrain = xgb.DMatrix(xtrain, label=y_train)
dtest = xgb.DMatrix(xtest)
##########################################################################################

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con =2
    x =preds-labels
    grad =con*x / (np.abs(x)+con)
    hess =con**2 / (np.abs(x)+con)**2
    return grad, hess


xgb_params = {
    'min_child_weight': 1.565476680604359849e+00,
    'eta': 0.001,
    'colsample_bytree': 3.772917715015777218e-01,
    'max_depth': 10,
    'subsample': 7.100797852877329674e-01,
    'alpha': 1.391915270438097929e+00,
    'gamma': 2.981406304821155206e+00,
    'silent': 1,
    'verbose_eval': True,
    'seed': 6174,
}

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y) - SHIFT, np.exp(yhat) - SHIFT)

res = xgb.cv(xgb_params, dtrain, num_boost_round=9999999,
             nfold=5,
             seed=SEED,
             stratified=False, obj=logregobj,
             early_stopping_rounds=2000,
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
    X_train, X_test, y_train, y_test = xtrain[tr], xtrain[te], label[tr], label[te]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    clf = xgb.train(xgb_params, dtrain, best_nrounds, obj=logregobj)
    dtest = xgb.DMatrix(X_test)
    pred = np.exp(clf.predict(dtest)) - SHIFT
    tmp = pd.DataFrame(pred, columns=sample.columns[1:])
    submission.iloc[te[0],0] = pred
    score[i]= mean_absolute_error(np.exp(y_test) - SHIFT, pred)
    print(score[i])
    i+=1

print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))
print(mean_absolute_error(np.exp(label) - SHIFT,submission.values))

submission.to_csv("XGB_retrain_2.csv", index_label='id')
########################################################################################
