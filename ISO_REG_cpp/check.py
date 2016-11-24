import numpy as np
import pandas as pd

train = pd.read_csv('../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
MIN_VALUE = np.min(train['loss'])

retrain = pd.read_csv('./cache/isotrain.csv', index_col=0)
test = pd.read_csv('./cache/isotest.csv', index_col=0)

before_test = pd.read_csv('./cache/testsubmission.csv', index_col=0)
