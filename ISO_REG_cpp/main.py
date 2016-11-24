import numpy as np
import pandas as pd

train = pd.read_csv('../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
MIN_VALUE = np.min(train['loss'])

retrain = pd.read_csv('../COMBINE/ver_58/retrain.csv', index_col=0)
retrain['prediction'] = retrain['loss']
del retrain['loss']
retrain.index = train.index
retrain['loss'] = train['loss']

test = pd.read_csv('../COMBINE/ver_58/pred_retrain.csv', index_col=0)
test['prediction'] = test['loss']
del test['loss']

retrain.to_csv("./cache/trainsubmission.csv")
test.to_csv("./cache/testsubmission.csv")
