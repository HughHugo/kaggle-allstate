import pandas as pd
import os

layer_1_path = '../../cache/layer_1'

# train
ET_train = pd.read_csv(os.path.join(layer_1_path, 'extraTree_retrain.csv'))
ET_tfidf_train = pd.read_csv(os.path.join(layer_1_path, "extraTree_tfidf_retrain.csv"))
xgb_train = pd.read_csv(os.path.join(layer_1_path, "xgb_retrain.csv"))
xgb_tfidf_train = pd.read_csv(os.path.join(layer_1_path, "xgb_tfidf_retrain.csv"))
rf_train = pd.read_csv(os.path.join(layer_1_path, "rf_retrain.csv"))
rf_tfidf_train = pd.read_csv(os.path.join(layer_1_path,"rf_tfidf_retrain.csv"))

assert ET_train.shape[0] == 188318
assert ET_tfidf_train.shape[0] == 188318
assert xgb_train.shape[0] == 188318
assert xgb_tfidf_train.shape[0] == 188318
assert rf_train.shape[0] == 188318
assert rf_tfidf_train.shape[0] == 188318



# test
ET_test = pd.read_csv(os.path.join(layer_1_path, "extraTree.csv"))
ET_tfidf_test = pd.read_csv(os.path.join(layer_1_path, "extraTree_tfidf.csv"))
xgb_test = pd.read_csv(os.path.join(layer_1_path, "xgb.csv"))
xgb_tfidf_test = pd.read_csv(os.path.join(layer_1_path, "xgb_tfidf.csv"))
rf_test = pd.read_csv(os.path.join(layer_1_path, "rf.csv"))
rf_tfidf_test = pd.read_csv(os.path.join(layer_1_path,"rf_tfidf.csv"))

assert ET_test.shape[0] == 125546
assert ET_tfidf_test.shape[0] == 125546
assert xgb_test.shape[0] == 125546
assert xgb_tfidf_test.shape[0] == 125546
assert rf_test.shape[0] == 125546
assert rf_tfidf_test.shape[0] == 125546
