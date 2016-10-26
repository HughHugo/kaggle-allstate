import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

path = '../../input/'
print("read training data")
train = pd.read_csv(path+"train.csv")
print("read test data")
test  = pd.read_csv(path+"test.csv")

names_cat = ['cat' + str(i+1) for i in range(116)]
for i in names_cat:
    le = LabelEncoder()
    le.fit(np.concatenate([train[i].values, test[i].values]))
    train[i] = le.transform(train[i])
    test[i] = le.transform(test[i])

layer_0_path = '../../cache/layer_0/'
train_tsne = pd.read_csv(layer_0_path+'train_tsne.csv')
train = train.join(train_tsne)

test_tsne = pd.read_csv(layer_0_path+'test_tsne.csv')
test = test.join(test_tsne)


layer_1_path = '../../cache/layer_1'
# train
ET_train = pd.read_csv(os.path.join(layer_1_path, 'extraTree_retrain.csv'))
ET_train.columns = ['id', 'ET']
ET_tfidf_train = pd.read_csv(os.path.join(layer_1_path, "extraTree_tfidf_retrain.csv"))
ET_tfidf_train.columns = ['id', 'ET_tfidf']
xgb_train = pd.read_csv(os.path.join(layer_1_path, "xgb_retrain.csv"))
xgb_train.columns = ['id', 'xgb']
xgb_tfidf_train = pd.read_csv(os.path.join(layer_1_path, "xgb_tfidf_retrain.csv"))
xgb_tfidf_train.columns = ['id', 'xgb_tfidf']
rf_train = pd.read_csv(os.path.join(layer_1_path, "rf_retrain.csv"))
rf_train.columns = ['id', 'rf']
rf_tfidf_train = pd.read_csv(os.path.join(layer_1_path,"rf_tfidf_retrain.csv"))
rf_tfidf_train.columns = ['id', 'rf_tfidf']

assert ET_train.shape[0] == 188318
assert ET_tfidf_train.shape[0] == 188318
assert xgb_train.shape[0] == 188318
assert xgb_tfidf_train.shape[0] == 188318
assert rf_train.shape[0] == 188318
assert rf_tfidf_train.shape[0] == 188318

train = train.merge(ET_train, how='left')
train = train.merge(ET_tfidf_train, how='left')
train = train.merge(xgb_train, how='left')
train = train.merge(xgb_tfidf_train, how='left')
train = train.merge(rf_train, how='left')
train = train.merge(rf_tfidf_train, how='left')

layer_1_merge_path = '../../cache/layer_1_merge/'
train.to_csv(layer_1_merge_path+"layer_1_train.csv",index_label='id')

# test
ET_test = pd.read_csv(os.path.join(layer_1_path, "extraTree.csv"))
ET_test.columns = ['id', 'ET']
ET_tfidf_test = pd.read_csv(os.path.join(layer_1_path, "extraTree_tfidf.csv"))
ET_tfidf_test.columns = ['id', 'ET_tfidf']
xgb_test = pd.read_csv(os.path.join(layer_1_path, "xgb.csv"))
xgb_test.columns = ['id', 'xgb']
xgb_tfidf_test = pd.read_csv(os.path.join(layer_1_path, "xgb_tfidf.csv"))
xgb_tfidf_test.columns = ['id', 'xgb_tfidf']
rf_test = pd.read_csv(os.path.join(layer_1_path, "rf.csv"))
rf_test.columns = ['id', 'rf']
rf_tfidf_test = pd.read_csv(os.path.join(layer_1_path,"rf_tfidf.csv"))
rf_tfidf_test.columns = ['id', 'rf_tfidf']

assert ET_test.shape[0] == 125546
assert ET_tfidf_test.shape[0] == 125546
assert xgb_test.shape[0] == 125546
assert xgb_tfidf_test.shape[0] == 125546
assert rf_test.shape[0] == 125546
assert rf_tfidf_test.shape[0] == 125546

test = test.merge(ET_test, how='left')
test = test.merge(ET_tfidf_test, how='left')
test = test.merge(xgb_test, how='left')
test = test.merge(xgb_tfidf_test, how='left')
test = test.merge(rf_test, how='left')
test = test.merge(rf_tfidf_test, how='left')

layer_1_merge_path = '../../cache/layer_1_merge/'
test.to_csv(layer_1_merge_path+"layer_1_test.csv",index_label='id')
