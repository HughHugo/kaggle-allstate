# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

pred_1 = pd.read_csv('../NN_1/NN_1.csv', index_col=0)
pred_2 = pd.read_csv('../XGB_2/XGB_2.csv', index_col=0)

print np.mean(pred_1)
print np.mean(pred_2)
pred = (pred_1 + pred_2)/2.0
pred.to_csv("pred.csv", index_label='id')
