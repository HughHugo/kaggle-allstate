import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, BaggingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

path = '../../input/'
cache_path = '../../cache/layer_0/'
cache_path_output = '../../cache/layer_1/'
print("read training data")
train = pd.read_csv(path+"train.csv")

#stack index
skf_table = pd.read_csv('../../cache/stack_index.csv')
skf_table = pd.merge(train, skf_table, on='id')
skf = []
nfold = 10
for i in range(nfold):
    skf += [[[(skf_table['stack_index']!=i).values],[(skf_table['stack_index']==i).values]]]

print("read test data")
test  = pd.read_csv(path+"test.csv")
ID = test['id']
del test['id']

names_cat = ['cat' + str(i+1) for i in range(116)]
for i in names_cat:
    print i
    le = LabelEncoder()
    le.fit(np.concatenate([train[i].values, test[i].values]))
    train[i] = le.transform(train[i])
    test[i] = le.transform(test[i])


label = np.log(train['loss'].values)
trainID = train['id']
del train['id']
del train['loss']
tsne = pd.read_csv(cache_path+'train_tsne.csv')
train = train.join(tsne)

dtrain = xgb.DMatrix(train.values, label=label)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.1,
    'objective': 'reg:linear',
    'max_depth': 3,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
}

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

res = xgb.cv(xgb_params, dtrain, num_boost_round=7500, nfold=4, seed=6174, stratified=False,
             early_stopping_rounds=50, verbose_eval=10, show_stdv=True, feval=xg_eval_mae, maximize=False)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]
print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

clf = xgb.train(xgb_params, dtrain, best_nrounds)

tsne = pd.read_csv(cache_path+'test_tsne.csv')
test = test.join(tsne)

dtest = xgb.DMatrix(test.values)

clf_probs = np.exp(clf.predict(dtest))
sample = pd.read_csv(path+'sample_submission.csv')
print("writing submission data")
submission = pd.DataFrame(clf_probs, index=ID, columns=sample.columns[1:])
submission.to_csv(cache_path_output+"xgb.csv",index_label='id')

# retrain

sample = pd.read_csv(path+'sample_submission.csv')
submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
score = np.zeros(nfold)
i=0
for tr, te in skf:
    X_train, X_test, y_train, y_test = train.values[tr], train.values[te], label[tr], label[te]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    clf = xgb.train(xgb_params, dtrain, best_nrounds)
    dtest = xgb.DMatrix(X_test)
    pred = np.exp(clf.predict(dtest))
    tmp = pd.DataFrame(pred, columns=sample.columns[1:])
    submission.iloc[te[0],0] = pred
    score[i]= mean_absolute_error(np.exp(y_test), pred)
    print(score[i])
    i+=1

print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))


print(mean_absolute_error(np.exp(label),submission.values))
submission.to_csv(cache_path_output+"xgb_retrain.csv",index_label='id')
