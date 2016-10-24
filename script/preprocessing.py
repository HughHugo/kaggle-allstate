print("Initialize libraries")
import os
import gc
import pandas as pd
import sys
import numpy as np
import scipy as sp
import copy
from sklearn.preprocessing import LabelEncoder, StandardScaler

#------------------------------------------------ Read data from source files ------------------------------------

seed = 6174
np.random.seed(seed)
datadir = '../input'
cache_dir = '../cache'
df_train = pd.read_csv(os.path.join(datadir,'train.csv'))
train_id = df_train['id']
df_test = pd.read_csv(os.path.join(datadir,'test.csv'))
test_id = df_test['id']

Y = df_train["loss"].values
del df_train['loss']
df_train =  pd.concat([df_train, df_test])
df_train = df_train.sort_values(['id'], ascending=[1])
df_train = df_train.reset_index(drop =True)


#=============== Feature Engineering ===============#
names_cat = ['cat' + str(i+1) for i in range(116)]
names_cont = ['cont' + str(i+1) for i in range(14)]
load_cache = False

if not load_cache:
    df_train_vector = None
    df_train_value_vector = None

    for i in names_cat:
        print i
        print i[3:]
        if int(i[3:]) <= 72:
            print "Case 1"
            #=============== Feature 1 ===============#
            tmp = copy.deepcopy(df_train[['id',i]])

            tmp_index = tmp[i] != tmp[i].mode()[0]
            tmp = tmp.loc[tmp_index,:]

            tmp.loc[:,i] = tmp[i] + '_' + i
            tmp_value = np.ones(len(tmp))
            tmp.columns = ['id', 'feature']

            assert tmp.shape[0] == len(tmp_value)

            # Save Feature 1
            if df_train_vector is None:
                df_train_vector = tmp
                df_train_value_vector = tmp_value
            else:
                df_train_vector = pd.concat([df_train_vector, tmp])
                df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

        elif int(i[3:]) <= 108:
            #continue
            print "Case 2"
            #=============== Feature 1 ===============#
            tmp = copy.deepcopy(df_train[['id',i]])

            tmp_index = tmp[i] != tmp[i].mode()[0]
            tmp = tmp.loc[tmp_index,:]

            tmp.loc[:,i] = tmp[i] + '_' + i
            tmp_value = np.ones(len(tmp))
            tmp.columns = ['id', 'feature']
            assert tmp.shape[0] == len(tmp_value)

            df_train_vector = pd.concat([df_train_vector, tmp])
            df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

            #=============== Feature 1 - Continuous ===============#
            tmp = copy.deepcopy(df_train[['id',i]])

            le = LabelEncoder()
            tmp_value = le.fit_transform(tmp[i])
            scaler = StandardScaler()
            tmp_value = scaler.fit_transform(tmp_value)

            tmp.loc[:,i] = tmp[i] + '_' + i
            tmp.columns = ['id', 'feature']
            assert tmp.shape[0] == len(tmp_value)

            df_train_vector = pd.concat([df_train_vector, tmp])
            df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

        else:
            print "Case 3"
            #=============== Feature 1 ===============#
            tmp = copy.deepcopy(df_train[['id',i]])

            tmp_index = tmp[i] != tmp[i].mode()[0]
            tmp = tmp.loc[tmp_index,:]

            tmp.loc[:,i] = tmp[i] + '_' + i
            tmp_value = np.ones(len(tmp))
            tmp.columns = ['id', 'feature']
            assert tmp.shape[0] == len(tmp_value)

            df_train_vector = pd.concat([df_train_vector, tmp])
            df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

            #=============== Feature 1 - Continuous ===============#
            tmp = copy.deepcopy(df_train[['id',i]])

            le = LabelEncoder()
            tmp_value = le.fit_transform(tmp[i])
            scaler = StandardScaler()
            tmp_value = scaler.fit_transform(tmp_value)

            tmp.loc[:,i] = tmp[i] + '_' + i
            tmp.columns = ['id', 'feature']
            assert tmp.shape[0] == len(tmp_value)

            df_train_vector = pd.concat([df_train_vector, tmp])
            df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

            #=============== Feature 4 - String len ===============#
            tmp = copy.deepcopy(df_train[['id',i]])
            tmp_index = tmp[i].apply(len) == 1
            if sum(tmp_index)>0:
                tmp = tmp.loc[tmp_index,:]

                le = LabelEncoder()
                tmp_value = le.fit_transform(tmp[i])
                scaler = StandardScaler()
                tmp_value = scaler.fit_transform(tmp_value)

                tmp.loc[:,i] = tmp[i] + '_' + i
                tmp.columns = ['id', 'feature']

                tmp['feature'] = tmp['feature'] + 'string_len_1'

                assert tmp.shape[0] == len(tmp_value)

                df_train_vector = pd.concat([df_train_vector, tmp])
                df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

            tmp = copy.deepcopy(df_train[['id',i]])
            tmp_index = tmp[i].apply(len) == 2
            if sum(tmp_index)>0:
                tmp = tmp.loc[tmp_index,:]
                le = LabelEncoder()
                tmp_value = le.fit_transform(tmp[i])
                scaler = StandardScaler()
                tmp_value = scaler.fit_transform(tmp_value)

                tmp.loc[:,i] = tmp[i] + '_' + i
                tmp.columns = ['id', 'feature']

                tmp['feature'] = tmp['feature'] + 'string_len_2'

                assert tmp.shape[0] == len(tmp_value)

                df_train_vector = pd.concat([df_train_vector, tmp])
                df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

        print df_train_vector.shape
        print len(df_train_value_vector)

    df_train_vector = df_train_vector.reset_index(drop =True)


    #=============== Feature Engineering - Continuous variable ===============#
    for i in names_cont:
        print i
        tmp = df_train[['id',i]]
        tmp.loc[:,i] = i
        tmp_value = df_train[i].values
        scaler = StandardScaler()
        tmp_value = scaler.fit_transform(tmp_value)
        tmp.columns = ['id', 'feature']

        assert tmp.shape[0] == len(tmp_value)

        df_train_vector = pd.concat([df_train_vector, tmp])
        df_train_value_vector = np.concatenate((df_train_value_vector, tmp_value))

        print df_train_vector.shape
        print len(df_train_value_vector)

    df_train_vector = df_train_vector.reset_index(drop =True)

    #=============== Save cache ===============#
    df_train_vector.to_csv(os.path.join(cache_dir,'df_x.csv'), index=False)
    np.save(os.path.join(cache_dir,'df_y'), df_train_value_vector)
else:
    df_train_vector = pd.read_csv(os.path.join(cache_dir,'df_x.csv'))
    df_train_value_vector = np.load(os.path.join(cache_dir,'df_y.npy'))
