import pandas as pd

tmp_1 = pd.read_csv("XGB_retrain_2__1.csv", index_col=0).fillna(0)
tmp_2 = pd.read_csv("XGB_retrain_2__2.csv", index_col=0).fillna(0)
tmp_3 = pd.read_csv("XGB_retrain_2__3.csv", index_col=0).fillna(0)
tmp_4 = pd.read_csv("XGB_retrain_2__4.csv", index_col=0).fillna(0)
tmp_5 = pd.read_csv("XGB_retrain_2__5.csv", index_col=0).fillna(0)
tmp_6 = pd.read_csv("XGB_retrain_2__6.csv", index_col=0).fillna(0)
tmp_7 = pd.read_csv("XGB_retrain_2__7.csv", index_col=0).fillna(0)
tmp_8 = pd.read_csv("XGB_retrain_2__8.csv", index_col=0).fillna(0)
tmp_9 = pd.read_csv("XGB_retrain_2__9.csv", index_col=0).fillna(0)
tmp_10 = pd.read_csv("XGB_retrain_2__10.csv", index_col=0).fillna(0)

tmp = tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5 + tmp_6 + tmp_7 + tmp_8 + tmp_9 + tmp_10

from sklearn.metrics import mean_absolute_error
train = pd.read_csv("../input/train.csv")
print mean_absolute_error(train['loss'], tmp['loss'])

tmp.to_csv("XGB_retrain_2.csv")
