
# Bag of apps categories
# Bag of labels categories
# Include phone brand and model device
import copy

print("Initialize libraries")
from sklearn.preprocessing import normalize
import pandas as pd
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.cluster import DBSCAN
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import ensemble
from sklearn.decomposition import PCA
import os
import gc
from scipy import sparse
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from keras.callbacks import EarlyStopping

#------------------------------------------------- Write functions ----------------------------------------

def rstr(df): return df.dtypes, df.head(3) ,df.apply(lambda x: [x.unique()]), df.apply(lambda x: [len(x.unique())]),df.shape

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

#------------------------------------------------ Read data from source files ------------------------------------

seed = 6174
np.random.seed(seed)
datadir = '../input'
df_train = pd.read_csv(os.path.join(datadir,'train.csv'))
train_id = df_train['id']
df_test = pd.read_csv(os.path.join(datadir,'test.csv'))
test_id = df_test['id']

Y = df_train["loss"].values
del df_train['loss']
df_train =  pd.concat([df_train, df_test])
df_train = df_train.reset_index(drop =True)


names_cat = ['cat' + str(i+1) for i in range(116)]
names_cont = ['cont' + str(i+1) for i in range(14)]

df_train_vector = None
df_train_value_vector = None

for i in names_cat:
    print i
    tmp = df_train[['id',i]]
    tmp.loc[:,i] = tmp[i] + '_' + i
    tmp_value = np.ones(len(tmp))
    tmp.columns = ['id', 'feature']
    if df_train_vector is None:
        df_train_vector = tmp
        df_train_value_vector = tmp_value
    else:
        df_train_vector = pd.concat([df_train_vector, tmp])
        df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

df_train_vector = df_train_vector.reset_index(drop =True)

for i in names_cont:
    print i
    tmp = df_train[['id',i]]
    tmp.loc[:,i] = i
    tmp_value = np.array(df_train[i])
    tmp.columns = ['id', 'feature']

    df_train_vector = pd.concat([df_train_vector, tmp])
    df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

df_train_vector = df_train_vector.reset_index(drop =True)

id_ind = True
if id_ind:
    tmp = df_train[['id',i]]
    tmp.loc[:,i] = 'id_feature'
    tmp_value = np.array(df_train['id'])
    tmp.columns = ['id', 'feature']

    df_train_vector = pd.concat([df_train_vector, tmp])
    df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

df_train_vector = df_train_vector.reset_index(drop =True)

FLS = df_train_vector
data = df_train_value_vector


print("# Create Sparse features")

device_ids = FLS["id"].unique()
feature_cs = FLS["feature"].unique()

dec = LabelEncoder().fit(FLS["id"])
row = dec.transform(FLS["id"])
col = LabelEncoder().fit_transform(FLS["feature"])
sparse_matrix = sparse.csr_matrix(
    (data, (row, col)), shape=(len(device_ids), len(feature_cs)))
sparse_matrix.shape
sys.getsizeof(sparse_matrix)
sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0]
sparse_matrix = normalize(sparse_matrix)
print("# Sparse matrix done")



print("# Split data")
train_row = dec.transform(train_id)
train_sp = sparse_matrix[train_row, :]

test_row = dec.transform(test_id)
test_sp = sparse_matrix[test_row, :]

X_train, X_val, y_train, y_val = train_test_split(
    train_sp, Y, train_size=0.99, random_state=6174)

##################
#   Feature Sel
##################
print("# Feature Selection")
print("# Num of Rows: ", X_train.shape[0])
print("# Num of Features: ", X_train.shape[1])

##################
#  Build Model
##################


#act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mae', optimizer='rmsprop')   #logloss
    return model

model=baseline_model()

# parameters
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=50,
                               mode="min")

fit= model.fit_generator(generator=batch_generator(X_train, y_train, 1024, True),
                         nb_epoch=100000,
                         samples_per_epoch=100000,
                         validation_data = (X_val.todense(), y_val),
                         verbose=1,
                         callbacks=[early_stopping]
                         )



# evaluate the model
scores_val = model.predict_generator(generator=batch_generatorp(X_val, 1024, False), val_samples=X_val.shape[0])
print('mae val {}'.format(mean_absolute_error(y_val, scores_val)))

print("# Final prediction")
scores = model.predict_generator(generator=batch_generatorp(test_sp, 1024, False), val_samples=test_sp.shape[0])
result = pd.DataFrame(scores)
result.columns = ['loss']
result["id"] = test_id
print(result.head(1))
result = result.set_index("id")

#result.to_csv('./sub_bagofapps7_keras_10_50_pt2_10epoch.csv', index=True, index_label='device_id')
#Drop out 0.2
#Validation 2.3017
result.to_csv('../sub/sub_2.csv', index=True, index_label='id')


print("Done")
