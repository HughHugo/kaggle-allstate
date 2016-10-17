print("Initialize libraries")
import os
import gc
import pandas as pd
import sys
import numpy as np
import scipy as sp
import copy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import rankdata

#------------------------------------------------ Read data from source files ------------------------------------

seed = 6174
np.random.seed(seed)
datadir = '../input'
cache_dir = '../cache'
df_train = pd.read_csv(os.path.join(datadir,'train.csv'))
Y = df_train['loss']

stack_index = rankdata(Y, method='ordinal')
stack_index = stack_index % 10

df_train['stack_index'] = stack_index
df_stack_index = df_train.loc[:,['id', 'stack_index']]

df_stack_index.to_csv(os.path.join(cache_dir,'stack_index.csv'), index=False)
