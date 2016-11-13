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

train = pd.read_csv('../../input/train.csv')
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

scaler = StandardScaler()
df_retrain = scaler.fit_transform(df_retrain)

SHIFT = 200
SEED = 6174
df_retrain_y = train['loss'].values
#df_retrain.columns = ["f_" + str(i) for i in range(20)]
y_train = df_retrain_y
## neural net

def nn_model():
    model = Sequential()

    model.add(Dense(100, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(25, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)


skf_table = pd.read_csv('../../cache/stack_index.csv')
skf_table = pd.merge(train, skf_table, on='id')
skf = []
nfold = 10
for i in range(nfold):
    skf += [[(skf_table['stack_index']!=i).values, (skf_table['stack_index']==i).values]]
print skf
print len(skf)
folds = skf
nfolds = nfold



## train models
i = 0
nbags = 5
nepochs = 75
xtrain=np.array(df_retrain)
y=df_retrain_y
pred_oob = np.zeros(xtrain.shape[0])

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=3,
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
    model = nn_model()
    model.fit(xtr, ytr, nb_epoch=200, batch_size=128, validation_data = (xte, yte), verbose = 1)
