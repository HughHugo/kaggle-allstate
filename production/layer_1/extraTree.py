import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, BaggingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

path = '../../input/'
cache_path = '../../cache/layer_0/'
cache_path_output = '../../cache/layer_1/'
print("read training data")
train = pd.read_csv(path+"train.csv")
print("read test data")
test  = pd.read_csv(path+"test.csv")
ID = test['id']
del test['id']

#stack index
skf_table = pd.read_csv('../../cache/stack_index.csv')
skf_table = pd.merge(train, skf_table, on='id')
skf = []
nfold = 10
for i in range(nfold):
    skf += [[[(skf_table['stack_index']!=i).values],[(skf_table['stack_index']==i).values]]]

names_cat = ['cat' + str(i+1) for i in range(116)]
for i in names_cat:
    print i
    le = LabelEncoder()
    le.fit(np.concatenate([train[i].values, test[i].values]))
    train[i] = le.transform(train[i])
    test[i] = le.transform(test[i])

label = np.log(train['loss'].values)
trainID = train['id']
del train['id']
del train['loss']
tsne = pd.read_csv(cache_path+'train_tsne.csv')
train = train.join(tsne)

clf = ExtraTreesRegressor(n_jobs=-1, criterion='mse', n_estimators=300, verbose=3, random_state=6174)
clf.fit(train.values, label)

tsne = pd.read_csv(cache_path+'test_tsne.csv')
test = test.join(tsne)

clf_probs = np.exp(clf.predict(test.values))
sample = pd.read_csv(path+'sample_submission.csv')
print("writing submission data")
submission = pd.DataFrame(clf_probs, index=ID, columns=sample.columns[1:])
submission.to_csv(cache_path_output+"extraTree.csv",index_label='id')

# retrain

sample = pd.read_csv(path+'sample_submission.csv')
submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
score = np.zeros(nfold)
i=0
for tr, te in skf:
	X_train, X_test, y_train, y_test = train.values[tr], train.values[te], label[tr], label[te]
	clf = ExtraTreesRegressor(n_jobs=-1, criterion='mse', n_estimators=300, verbose=3, random_state=6174)
	clf.fit(X_train, y_train)
	pred = np.exp(clf.predict(X_test))
	tmp = pd.DataFrame(pred, columns=sample.columns[1:])
	submission.iloc[te[0],0] = pred
	score[i]= mean_absolute_error(np.exp(y_test), pred)
	print(score[i])
	i+=1

print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))


print(mean_absolute_error(label,submission.values))
submission.to_csv(cache_path_output+"extraTree_retrain.csv",index_label='id')
