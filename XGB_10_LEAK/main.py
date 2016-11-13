# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

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

#train.drop([ID, TARGET], axis=1, inplace=True)
#test.drop([ID], axis=1, inplace=True)

print("{},{}".format(train.shape, test.shape))

ntrain = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)

features = train.columns

### remeber the order
train_test[ID]=pd.Categorical(train_test[ID], train_test[ID].values.tolist())

### factorize
cats = [feat for feat in train.columns if 'cat' in feat]
for cat in cats:
    sorting_list=np.unique(sorted(train_test[cat],key=lambda x:(str.lower(x),x)))
    train_test[cat]=pd.Categorical(train_test[cat], sorting_list)
    train_test=train_test.sort_values(cat)
    train_test[cat] = pd.factorize(train_test[cat], sort=True)[0]

### reorder
train_test=train_test.sort_values(ID)
#gc.collect()

print(train_test.head())

#### preprocessing
train_test["cont1"] = np.sqrt(preprocessing.minmax_scale(train_test["cont1"]))
train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))

train_test["cont6"] = np.log(preprocessing.minmax_scale(train_test["cont6"])+0000.1)
train_test["cont7"] = np.log(preprocessing.minmax_scale(train_test["cont7"])+0000.1)
train_test["cont9"] = np.log(preprocessing.minmax_scale(train_test["cont9"])+0000.1)
train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"])+0000.1)
train_test["cont14"]=(np.maximum(train_test["cont14"]-0.179722,0)/0.665122)**0.25

####

train_test.drop([ID, TARGET], axis=1, inplace=True)


x_train = np.array(train_test.iloc[:ntrain,:])
x_test = np.array(train_test.iloc[ntrain:,:])

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

#import time
#time.sleep(60)

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con =2
    x =preds-labels
    grad =con*x / (np.abs(x)+con)
    hess =con**2 / (np.abs(x)+con)**2
    return grad, hess

xgb_params = {
    'colsample_bytree': 0.3085,
    'subsample': 0.9930,
    'eta': 0.001,
    'gamma': 0.5290,
    'booster' :  'gbtree',
    'objective': 'reg:linear',
    'max_depth': 7,
    'min_child_weight': 4.2923,
    'verbose_eval': True,
    'seed': 6174,
    'silent': 1
}

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y) - SHIFT, np.exp(yhat) - SHIFT)

#res = xgb.cv(xgb_params, dtrain, num_boost_round=9999999,
#             nfold=4,
#             seed=SEED,
            #  stratified=False, obj=logregobj,
            #  early_stopping_rounds=300,
            #  verbose_eval=10,
            #  show_stdv=True,
            #  feval=xg_eval_mae,
            #  maximize=False)

#best_nrounds = res.shape[0] - 1
#cv_mean = res.iloc[-1, 0]
#cv_std = res.iloc[-1, 1]
#print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

#gbdt = xgb.train(xgb_params, dtrain, best_nrounds, obj=logregobj)

submission_ALL = pd.read_csv(SUBMISSION_FILE)
#submission_ALL.iloc[:, 1] = np.exp(gbdt.predict(dtest)) - SHIFT
#submission.to_csv('XGB_2.csv', index=None)

########################################################################################
sample = pd.read_csv('../input/sample_submission.csv')
submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
score = np.zeros(nfold)
i=0
best_nrounds = 1000000
for tr, te in skf:
    tr = np.where(tr)
    te = np.where(te)
    X_train, X_test, y_train, y_test = x_train[tr], x_train[te], label[tr], label[te]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)
    watchlist  = [ (dtrain,'train'),(dval,'eval')]
    clf = xgb.train(xgb_params,
                    dtrain,
                    best_nrounds,
                    evals=watchlist,
                    obj=logregobj,
                    feval=xg_eval_mae,
                    early_stopping_rounds=1000,
                    verbose_eval=100,
                    maximize=False)
    pred_val = np.exp(clf.predict(dval)) - SHIFT
    pred_ALL = np.exp(clf.predict(dtest)) - SHIFT
    submission.iloc[te[0],0] = pred_val
    submission_ALL.iloc[:,1] = submission_ALL.iloc[:,1] + pred_ALL
    score[i]= mean_absolute_error(np.exp(y_test) - SHIFT, pred_val)
    print(score[i])
    i+=1


print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))
print(mean_absolute_error(np.exp(label) - SHIFT,submission.values))

submission.to_csv("XGB_retrain_3.csv", index_label='id')

submission_ALL.iloc[:,1] = submission_ALL.iloc[:,1]/10.
submission_ALL.to_csv("XGB_3.csv", index=None)

########################################################################################
