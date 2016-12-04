# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('../../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
# Main
pred_nn_1_retrain = pd.read_csv('../../NN_1/NN_retrain_1.csv', index_col=0)
pred_nn_2_retrain = pd.read_csv('../../NN_2/NN_retrain_2.csv', index_col=0)
pred_nn_3_retrain = pd.read_csv('../../NN_3/NN_retrain_3.csv', index_col=0)
pred_nn_4_retrain = pd.read_csv('../../NN_4/NN_retrain_4.csv', index_col=0)
pred_nn_5_retrain = pd.read_csv('../../NN_5/NN_retrain_5.csv', index_col=0)
pred_nn_6_retrain = pd.read_csv('../../NN_6/NN_retrain_6.csv', index_col=0)
pred_nn_1_fix_retrain = pd.read_csv('../../NN_1_fix/NN_retrain_1.csv', index_col=0)
pred_nn_2_fix_retrain = pd.read_csv('../../NN_2_fix/NN_retrain_2.csv', index_col=0)
pred_nn_3_fix_retrain = pd.read_csv('../../NN_3_fix/NN_retrain_3.csv', index_col=0)
pred_nn_4_fix_retrain = pd.read_csv('../../NN_4_fix/NN_retrain_4.csv', index_col=0)
pred_nn_5_fix_retrain = pd.read_csv('../../NN_5_fix/NN_retrain_5.csv', index_col=0)
pred_nn_6_fix_retrain = pd.read_csv('../../NN_6_fix/NN_retrain_6.csv', index_col=0)
pred_new_nn_1_retrain = pd.read_csv('../../NEW_NN_1/NN_retrain_1.csv', index_col=0)
pred_new_nn_2_retrain = pd.read_csv('../../NEW_NN_2/NN_retrain_2.csv', index_col=0)
pred_new_nn_3_retrain = pd.read_csv('../../NEW_NN_3/NN_retrain_3.csv', index_col=0)
pred_new_nn_3_retrain.loc[pred_new_nn_3_retrain['loss']>MAX_VALUE,:] = MAX_VALUE
pred_new_nn_4_retrain = pd.read_csv('../../NEW_NN_4/NN_retrain_4.csv', index_col=0)
pred_new_nn_4_retrain.loc[pred_new_nn_4_retrain['loss']>MAX_VALUE,:] = MAX_VALUE
pred_new_nn_1_65_retrain = pd.read_csv('../../NEW_NN_1_65/NN_retrain_1.csv', index_col=0)
#sorted(pred_new_nn_3_retrain[130000:150000].values, reverse=True)

pred_nn_1 = pd.read_csv('../../NN_1/NN_1.csv', index_col=0)
pred_nn_2 = pd.read_csv('../../NN_2/NN_2.csv', index_col=0)
pred_nn_3 = pd.read_csv('../../NN_3/NN_3.csv', index_col=0)
pred_nn_4 = pd.read_csv('../../NN_4/NN_4.csv', index_col=0)
pred_nn_5 = pd.read_csv('../../NN_5/NN_5.csv', index_col=0)
pred_nn_6 = pd.read_csv('../../NN_6/NN_6.csv', index_col=0)
pred_nn_1_fix = pd.read_csv('../../NN_1_fix/NN_1.csv', index_col=0)
pred_nn_2_fix = pd.read_csv('../../NN_2_fix/NN_2.csv', index_col=0)
pred_nn_3_fix = pd.read_csv('../../NN_3_fix/NN_3.csv', index_col=0)
pred_nn_4_fix = pd.read_csv('../../NN_4_fix/NN_4.csv', index_col=0)
pred_nn_5_fix = pd.read_csv('../../NN_5_fix/NN_5.csv', index_col=0)
pred_nn_6_fix = pd.read_csv('../../NN_6_fix/NN_6.csv', index_col=0)
pred_new_nn_1 = pd.read_csv('../../NEW_NN_1/NN_1.csv', index_col=0)
pred_new_nn_2 = pd.read_csv('../../NEW_NN_2/NN_2.csv', index_col=0)
pred_new_nn_3 = pd.read_csv('../../NEW_NN_3/NN_3.csv', index_col=0)
pred_new_nn_4 = pd.read_csv('../../NEW_NN_4/NN_4.csv', index_col=0)
pred_new_nn_3.loc[pred_new_nn_3['loss']>MAX_VALUE,:] = MAX_VALUE
pred_new_nn_4.loc[pred_new_nn_4['loss']>MAX_VALUE,:] = MAX_VALUE
pred_new_nn_1_65 = pd.read_csv('../../NEW_NN_1_65/NN_1.csv', index_col=0)

pred_xgb_1_retrain = pd.read_csv('../../XGB_1/XGB_retrain_1.csv', index_col=0)
pred_xgb_2_retrain = pd.read_csv('../../XGB_2/XGB_retrain_2.csv', index_col=0)
pred_xgb_3_retrain = pd.read_csv('../../XGB_3/XGB_retrain_3.csv', index_col=0)

pred_xgb_1 = pd.read_csv('../../XGB_1/XGB_1.csv', index_col=0)
pred_xgb_2 = pd.read_csv('../../XGB_2/XGB_2.csv', index_col=0)
pred_xgb_3 = pd.read_csv('../../XGB_3/XGB_3.csv', index_col=0)


print mean_absolute_error(train['loss'], pred_nn_1_retrain)
print mean_absolute_error(train['loss'], pred_nn_2_retrain)
print mean_absolute_error(train['loss'], pred_nn_3_retrain)
print mean_absolute_error(train['loss'], pred_nn_4_retrain)
print mean_absolute_error(train['loss'], pred_nn_5_retrain)
print mean_absolute_error(train['loss'], pred_nn_6_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_nn_1_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_2_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_3_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_4_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_5_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_6_fix_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_new_nn_1_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_2_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_3_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_4_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_new_nn_1_65_retrain)

df_retrain = pd.concat([
                           pred_nn_1_fix_retrain['loss'],    #1
                           pred_nn_2_fix_retrain['loss'],    #2
                           pred_nn_3_fix_retrain['loss'],    #3
                           pred_nn_4_fix_retrain['loss'],    #4
                           pred_nn_5_fix_retrain['loss'],    #5
                           pred_nn_6_fix_retrain['loss'],    #6
                           pred_new_nn_1_retrain['loss'],    #7
                           pred_new_nn_2_retrain['loss'],    #8
                           pred_new_nn_3_retrain['loss'],    #9
                           pred_new_nn_4_retrain['loss'],    #10
                           pred_new_nn_1_65_retrain['loss'], #11
                           pred_xgb_1_retrain['loss'],       #12
                           pred_xgb_2_retrain['loss'],       #13
                           pred_xgb_3_retrain['loss'],       #14
                           pred_nn_1_retrain['loss'],        #15
                           pred_nn_2_retrain['loss'],        #16
                           pred_nn_3_retrain['loss'],        #17
                           pred_nn_4_retrain['loss'],        #18
                           pred_nn_5_retrain['loss'],        #19
                           pred_nn_6_retrain['loss'],        #20
                           ], axis=1)

SHIFT = 200
SEED = 6174
df_retrain_y = np.log(train['loss'].values + SHIFT)
df_retrain.columns = ["f_" + str(i) for i in range(20)]

y_train = df_retrain_y

dtrain = xgb.DMatrix(df_retrain, label=y_train)


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con =2
    x =preds-labels
    grad =con*x / (np.abs(x)+con)
    hess =con**2 / (np.abs(x)+con)**2
    return grad, hess

xgb_params = {
    'min_child_weight': 1,
    'eta': 0.001,
    'colsample_bytree': 0.3,
    'max_depth': 8,
    'subsample': 0.6,
    #'alpha': 1,
    #'gamma': 1,
    'silent': 1,
    'verbose_eval': True,
    'seed': 6174,
}


def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y) - SHIFT, np.exp(yhat) - SHIFT)

res = xgb.cv(xgb_params, dtrain, num_boost_round=999999,
             nfold=10,
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

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)#obj=logregobj

df_test = pd.concat([
                           pred_nn_1_fix['loss'],    #1
                           pred_nn_2_fix['loss'],    #2
                           pred_nn_3_fix['loss'],    #3
                           pred_nn_4_fix['loss'],    #4
                           pred_nn_5_fix['loss'],    #5
                           pred_nn_6_fix['loss'],    #6
                           pred_new_nn_1['loss'],    #7
                           pred_new_nn_2['loss'],    #8
                           pred_new_nn_3['loss'],    #9
                           pred_new_nn_4['loss'],    #10
                           pred_new_nn_1_65['loss'], #11
                           pred_xgb_1['loss'],       #12
                           pred_xgb_2['loss'],       #13
                           pred_xgb_3['loss'],       #14
                           pred_nn_1['loss'],        #15
                           pred_nn_2['loss'],        #16
                           pred_nn_3['loss'],        #17
                           pred_nn_4['loss'],        #18
                           pred_nn_5['loss'],        #19
                           pred_nn_6['loss'],        #20
                           ], axis=1)

df_test.columns = ["f_" + str(i) for i in range(20)]
dtest= xgb.DMatrix(df_test)


tmp = np.exp(gbdt.predict(dtest))  - SHIFT
DATA_DIR = "../../input"
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = tmp
submission.to_csv('pred_retrain.csv', index=None)
