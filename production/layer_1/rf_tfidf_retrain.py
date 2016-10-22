import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

path = '../../input/'
cache_path = '../../cache/layer_0/'
cache_path_output = '../../cache/layer_1/'
print("read training data")
train = pd.read_csv(cache_path+"train_tfidf.csv")

#stack index
skf_table = pd.read_csv('../../cache/stack_index.csv')
skf_table = pd.merge(train, skf_table, on='id')
skf = []
nfold = 10
for i in range(nfold):
    skf += [[[(skf_table['stack_index']!=i).values],[(skf_table['stack_index']==i).values]]]

label = np.log(train['loss'].values)
trainID = train['id']
del train['id']
del train['loss']
tsne = pd.read_csv(cache_path+'tfidf_train_tsne.csv')
train = train.join(tsne)

clf = RandomForestRegressor(n_jobs=-1, criterion='mse', n_estimators=300, verbose=3, random_state=6174)
clf.fit(train.values, label)

print("read test data")
test  = pd.read_csv(cache_path+"test_tfidf.csv")
ID = test['id']
del test['id']

tsne = pd.read_csv(cache_path+'tfidf_test_tsne.csv')
test = test.join(tsne)

clf_probs = np.exp(clf.predict(test.values))
sample = pd.read_csv(path+'sample_submission.csv')
print("writing submission data")
submission = pd.DataFrame(clf_probs, index=ID, columns=sample.columns[1:])
submission.to_csv(cache_path_output+"rf_tfidf.csv",index_label='id')

# retrain

sample = pd.read_csv(path+'sample_submission.csv')
submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
score = np.zeros(nfold)
i=0
for tr, te in skf:
	X_train, X_test, y_train, y_test = train.values[tr], train.values[te], label[tr], label[te]
	clf = RandomForestRegressor(n_jobs=-1, criterion='mse', n_estimators=300, verbose=3, random_state=6174)
	clf.fit(X_train, y_train)
	pred = np.exp(clf.predict(X_test))
	tmp = pd.DataFrame(pred, columns=sample.columns[1:])
	submission.iloc[te[0],0] = pred
	score[i]= mean_absolute_error(np.exp(y_test), pred)
	print(score[i])
	i+=1

print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))


print(mean_absolute_error(label,submission.values))
submission.to_csv(path+"rf_tfidf_retrain.csv",index_label='id')
