import pandas as pd

tmp_1 = pd.read_csv("XGB_retrain_2__1.csv", index_col=0).fillna(0)
tmp_3 = pd.read_csv("XGB_retrain_2__3.csv", index_col=0).fillna(0)
tmp_5 = pd.read_csv("XGB_retrain_2__5.csv", index_col=0).fillna(0)
tmp_7 = pd.read_csv("XGB_retrain_2__7.csv", index_col=0).fillna(0)
tmp_9 = pd.read_csv("XGB_retrain_2__9.csv", index_col=0).fillna(0)

tmp = tmp_1 + tmp_3 + tmp_5 + tmp_7 + tmp_9

from sklearn.metrics import mean_absolute_error
train = pd.read_csv("../input/train.csv")
print mean_absolute_error(train['loss'], tmp['loss'])

tmp.to_csv("XGB_retrain_2.csv")
