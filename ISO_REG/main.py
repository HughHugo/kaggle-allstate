import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

train = pd.read_csv('../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
MIN_VALUE = np.min(train['loss'])

retrain = pd.read_csv('../COMBINE/ver_58/retrain.csv', index_col=0)
retrain.index = train.index

ir = IsotonicRegression(y_min = MIN_VALUE,
                        y_max = MAX_VALUE,
                        out_of_bounds = 'clip')

ir.fit( retrain['loss'].values, train['loss'].values)

test = pd.read_csv('../COMBINE/ver_58/pred_retrain.csv', index_col=0)

p_calibrated = ir.transform( test['loss'].values)


test['loss'] = p_calibrated
print np.max(test['loss'])
print np.min(test['loss'])

test.to_csv("./cache/pred_retrain.csv")
