import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import os
import ml_metrics as metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, BaggingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import random

path = '../../input/'
layer_0_path = '../../cache/layer_0/'
layer_1_path = '../../cache/layer_1/'
layer_1_merge_path = '../../cache/layer_1_merge/'
layer_2_path = '../../cache/layer_1_merge/'
print("read training data")
train = pd.read_csv(layer_1_merge_path+"layer_1_train.csv")
print("read test data")
test  = pd.read_csv(layer_1_merge_path+"layer_1_test.csv")

#stack index
skf_table = pd.read_csv('../../cache/stack_index.csv')
skf_table = pd.merge(train, skf_table, on='id')
skf = []
nfold = 10
for i in range(nfold):
    skf += [[[(skf_table['stack_index']!=i).values],[(skf_table['stack_index']==i).values]]]

ID = test['id']
del test['id']
label = np.log(train['loss'].values)
trainID = train['id']
del train['id']
del train['loss']

dtrain = xgb.DMatrix(train.values, label=label)
def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

if os.path.isfile("../../log/xgb_param.log"):
    print "1"
else:
    print "2"
    f=open("../../log/xgb_param.log", "w")
    f.write("booster,eta,max_depth,gamma,min_child_weight,max_delta_step,subsample,colsample_bytree,colsample_bylevel,lambda,best_nrounds,cv_mean,cv_sd")
    f.write("\n")
    f.close()

k=0
while True:
    print k
    k=k+1
    f=open("../../log/xgb_param.log", "a")

    xgb_params = {
        'booster': random.choice(['gblinear', 'gbtree']),
        'seed': 6174,
        'silent': 1,
        'eta': random.uniform(0.01, 0.05),
        'objective': 'reg:linear',
        'max_depth': random.randint(3, 9),
        'gamma': random.uniform(0, 1),
        'min_child_weight': random.uniform(0, 2),
        'max_delta_step': random.uniform(0, 5),
        'subsample': random.uniform(0.3, 1),
        'colsample_bytree': random.uniform(0.3, 1),
        'colsample_bylevel': random.uniform(0.3, 1),
        'lambda': random.uniform(0, 5),
        'eval_metric': 'mae'
    }

    res = xgb.cv(xgb_params,
                dtrain,
                num_boost_round=1000000,
                nfold=5,
                seed=6174,
                early_stopping_rounds=100,
                verbose_eval=10,
                show_stdv=True,
                feval=xg_eval_mae,
                maximize=False)

    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 0]
    cv_std = res.iloc[-1, 1]
    xgb_params['best_nrounds'] = best_nrounds
    xgb_params['cv_mean'] = cv_mean
    xgb_params['cv_std'] = cv_std
    f.write(str(xgb_params["booster"]) + ","+ str(xgb_params["eta"]) + "," +
         str(xgb_params["max_depth"]) + "," +
         str(xgb_params["gamma"]) + "," + str(xgb_params["min_child_weight"]) + "," +
         str(xgb_params["max_delta_step"]) + "," + str(xgb_params["subsample"]) + "," +
         str(xgb_params["colsample_bytree"]) + "," + str(xgb_params["colsample_bylevel"]) + "," +
         str(xgb_params["lambda"]) +  "," + str(xgb_params["best_nrounds"]) +  "," +
         str(xgb_params["cv_mean"]) +  "," + str(xgb_params["cv_std"]))
    f.write("\n")
    f.close()
