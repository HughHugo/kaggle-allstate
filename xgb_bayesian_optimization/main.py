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

##########################################################################################
ID = 'id'
TARGET = 'loss'
SEED = 6174
DATA_DIR = "../input"
SHIFT = 200

TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)


train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

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
        'eta': 0.1,
        'silent': 1,
        'verbose_eval': True,
        'seed': SEED
    }

    xgb_params['min_child_weight'] = float(min_child_weight)
    xgb_params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    xgb_params['max_depth'] = int(max_depth)
    xgb_params['subsample'] = max(min(subsample, 1), 0)
    xgb_params['gamma'] = max(gamma, 0)
    xgb_params['alpha'] = max(alpha, 0)


    cv_result = xgb.cv(xgb_params,
                             dtrain,
                             num_boost_round=20,
                             nfold=4,
                             seed=SEED,
                             stratified=False, obj=logregobj,
                             early_stopping_rounds=50,
                             verbose_eval=1,
                             show_stdv=True,
                             feval=xg_eval_mae,
                             maximize=False
                       )
    print (-cv_result['test-mae-mean'].values[-1] + 2000)/1000.
    return (-cv_result['test-mae-mean'].values[-1] + 2000)/1000.

xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (0.5, 1.5),
                                            'colsample_bytree': (0.3, 0.7),
                                            'max_depth': (10, 20),
                                            'subsample': (0.5, 1),
                                            'gamma': (0.5, 2),
                                            'alpha': (0, 2),
                                            })

num_iter = 100
init_points = 1
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
