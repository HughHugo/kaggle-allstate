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


SEED = 6174
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
y = np.log(train['loss'].values + SHIFT)
id_train = train['id'].values
id_test = test['id'].values

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

dtrain = xgb.DMatrix(xtrain, label=y)
dtest = xgb.DMatrix(xtest)
##########################################################################################

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con =2
    x =preds-labels
    grad =con*x / (np.abs(x)+con)
    hess =con**2 / (np.abs(x)+con)**2
    return grad, hess


def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y) - SHIFT, np.exp(yhat) - SHIFT)


def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma,
                 alpha):

    xgb_params = {
        'eta': 0.3,
        'silent': 1,
        'verbose_eval': True,
        'seed': SEED
    }

    xgb_params['min_child_weight'] = float(min_child_weight)
    xgb_params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    xgb_params['max_depth'] = int(round(max_depth))
    xgb_params['subsample'] = max(min(subsample, 1), 0)
    xgb_params['gamma'] = max(gamma, 0)
    xgb_params['alpha'] = max(alpha, 0)


    cv_result = xgb.cv(xgb_params,
                             dtrain,
                             num_boost_round=1000,
                             nfold=5,
                             seed=SEED,
                             stratified=False, obj=logregobj,
                             early_stopping_rounds=50,
                             verbose_eval=100,
                             show_stdv=True,
                             feval=xg_eval_mae,
                             maximize=False
                       )
    print (-cv_result['test-mae-mean'].values[-1] + 2000)/1000.
    return (-cv_result['test-mae-mean'].values[-1] + 2000)/1000.

xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (0.5, 2.0),
                                            'colsample_bytree': (0.3, 0.9),
                                            'max_depth': (10, 25),
                                            'subsample': (0.5, 1.0),
                                            'gamma': (0.5, 5.0),
                                            'alpha': (0.0, 5.0),
                                            })

num_iter = 1
init_points = 30
xgbBO.maximize(init_points=init_points, n_iter=num_iter)

while True:
    xgbBO.maximize(n_iter=num_iter)

    # Save .csv
    xgbBO.points_to_csv("xgb_bayes_opt.csv")

    # Save .pkl
    filehandler = open("xgb_bayes_opt.pkl","wb")
    pickle.dump(xgbBO,filehandler)
    filehandler.close()
    filehandler = open("xgb_bayes_opt.pkl",'rb')
    xgbBO = pickle.load(filehandler)
    filehandler.close()
