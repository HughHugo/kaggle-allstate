import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import itertools


shift = 200
SHIFT = 200
SEED = 6174
COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(
    ',')
################################################################################

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
################################################################################

def encode(charcode):
    r = 0
    if(type(charcode) is float):
        return np.nan
    else:
        ln = len(charcode)
        for i in range(ln):
            r += (ord(charcode[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
        return r


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + 0.7
    grad = 0.7 * x / (den)
    hess = 0.7 * 0.7 / (den * den)
    return grad, hess


def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y) - shift,
                                      np.exp(yhat) - shift)


def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    # compute skew and do Box-Cox transformation (Tilli)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain


if __name__ == "__main__":
    print('Started')
    directory = '../input/'
    train = pd.read_csv(directory + 'train.csv')
    test = pd.read_csv(directory + 'test.csv')
    numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
    cats = [x for x in train.columns[1:-1] if 'cat' in x]
    train_test, ntrain = mungeskewed(train, test, numeric_feats)

    # taken from Vladimir's script (https://www.kaggle.com/iglovikov/allstate-claims-severity/xgb-1114)
    for column in list(train.select_dtypes(include=['object']).columns):
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)


            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x


            train_test[column] = train_test[column].apply(lambda x: filter_cat(x), 1)

    # taken from Ali's script (https://www.kaggle.com/aliajouz/allstate-claims-severity/singel-model-lb-1117)
    train_test["cont1"] = np.sqrt(preprocessing.minmax_scale(train_test["cont1"]))
    train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
    train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
    train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
    train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
    train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
    train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))

    train_test["cont6"] = np.log(preprocessing.minmax_scale(train_test["cont6"]) + 0000.1)
    train_test["cont7"] = np.log(preprocessing.minmax_scale(train_test["cont7"]) + 0000.1)
    train_test["cont9"] = np.log(preprocessing.minmax_scale(train_test["cont9"]) + 0000.1)
    train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"]) + 0000.1)
    train_test["cont14"] = (np.maximum(train_test["cont14"] - 0.179722, 0) / 0.665122) ** 0.25

    np.random.seed(6174)
    for comb in itertools.combinations(COMB_FEATURE, 3):
        if np.random.uniform(0,1) < 0.9:
            print "#"
            continue
        feat = comb[0] + "_" + comb[1] + "_" + comb[2]
        train_test[feat] = train_test[comb[0]] + train_test[comb[1]] + train_test[comb[2]]
        train_test[feat] = train_test[feat].apply(encode)
        print(feat)


    cats = [x for x in train.columns[1:-1] if 'cat' in x]
    for col in cats:
        train_test[col] = train_test[col].apply(encode)
    train_test.loss = np.log(train_test.loss + shift)
    ss = StandardScaler()
    train_test[numeric_feats] = \
        ss.fit_transform(train_test[numeric_feats].values)
    train = train_test.iloc[:ntrain, :].copy()
    label = train.loss.values
    train.drop('loss', inplace=True, axis=1)
    test = train_test.iloc[ntrain:, :].copy()
    test.drop('loss', inplace=True, axis=1)
    del train_test

    xgb_params = {
        'seed': 6174,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.001,
        'objective': 'reg:linear',
        'max_depth': 12,
        'min_child_weight': 100,
        'alpha': 1,
        'gamma': 1,
        'booster': 'gbtree',
        'verbose_eval': True,
    }

    x_train = np.array(train)
    del train
    x_test = np.array(test)
    del test

    dtrain = xgb.DMatrix(x_train,
                          label=label)
    dtest = xgb.DMatrix(x_test)

    res = xgb.cv(xgb_params, dtrain, num_boost_round=250000,
             nfold=4,
             seed=SEED,
             stratified=False, obj=logregobj,
             early_stopping_rounds=1000,
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
        score[i]= mean_absolute_error(np.exp(y_test) - SHIFT, pred)
        print(score[i])
        i+=1

    print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))
    print(mean_absolute_error(np.exp(label) - SHIFT,submission.values))

    submission.to_csv("XGB_retrain_2.csv", index_label='id')
    ########################################################################################
