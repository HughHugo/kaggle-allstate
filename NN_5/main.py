# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(6174)
import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

## Batch generators ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

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
y = train['loss'].values
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

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(800, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(400, init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(200, init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.1))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)

## cv-folds
nfolds = nfold
folds = skf

## train models
i = 0
nbags = 5
nepochs = 1000
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               mode="min")

for inTr, inTe in folds:
    print inTr
    print inTe
    inTr = np.where(inTr)
    inTe = np.where(inTe)
    print inTr
    print inTe
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])

    # Find best iter
    tmp_iter = 0
    tmp_n_folds = 5
    tmp_folds = KFold(len(ytr), n_folds = tmp_n_folds, shuffle = True, random_state = 6174)
    for (cv_Tr, cv_Te) in tmp_folds:
        x_cv_tr = xtr[cv_Tr]
        y_cv_tr = ytr[cv_Tr]
        x_cv_te = xtr[cv_Te]
        y_cv_te = ytr[cv_Te]
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(x_cv_tr, y_cv_tr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  validation_data = (x_cv_te.todense(), y_cv_te),
                                  callbacks=[early_stopping],
                                  verbose = 0)
        tmp_iter = tmp_iter + fit.history['val_loss'].index(min(fit.history['val_loss'])) - 1
        print tmp_iter
    tmp_iter = int(round(tmp_iter/float(tmp_n_folds)))
    print tmp_iter

    # Train
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = tmp_iter,
                                  samples_per_epoch = xtr.shape[0],
                                  validation_data = (xte.todense(), yte),
                                  verbose = 0)
        pred += model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0]
        pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0]

    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(yte, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(y, pred_oob))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('NN_retrain_5.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('NN_5.csv', index = False)
