import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor

path = '../../input/'
cache_path = '../../cache/layer_0/'
cache_path_output = '../../cache/layer_1/'
print("read training data")
train = pd.read_csv(path+"train.csv")

#stack index
skf_table = pd.read_csv('../../cache/stack_index.csv')
skf_table = pd.merge(train, skf_table, on='id')
skf = []
nfold = 10
for i in range(nfold):
    skf += [[[(skf_table['stack_index']!=i).values],[(skf_table['stack_index']==i).values]]]

print("read test data")
test  = pd.read_csv(path+"test.csv")
ID = test['id']
del test['id']

names_cat = ['cat' + str(i+1) for i in range(116)]
for i in names_cat:
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

tsne = pd.read_csv(cache_path+'test_tsne.csv')
test = test.join(tsne)

combine = pd.concat([train, test])

train = (train - combine.mean())/(combine.max() - combine.min())
test = (test - combine.mean())/(combine.max() - combine.min())

for numk in range(10):
    NumK = numk+1
    clf = KNeighborsRegressor(n_neighbors=2**NumK,n_jobs=-1)
    clf.fit(train.values, label)

    clf_probs = np.exp(clf.predict(test.values))
    sample = pd.read_csv(path+'sample_submission.csv')
    print("writing submission data")
    submission = pd.DataFrame(clf_probs, index=ID, columns=sample.columns[1:])
    submission.to_csv(cache_path_output+"knn_%s.csv" %NumK, index_label='id')

    # retrain
    sample = pd.read_csv(path+'sample_submission.csv')
    submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
    score = np.zeros(nfold)
    i=0
    for tr, te in skf:
    	X_train, X_test, y_train, y_test = train.values[tr], train.values[te], label[tr], label[te]
    	clf = KNeighborsRegressor(n_neighbors=2**NumK, n_jobs=-1)
    	clf.fit(X_train, y_train)
    	pred = np.exp(clf.predict(X_test))
    	tmp = pd.DataFrame(pred, columns=sample.columns[1:])
    	submission.iloc[te[0],0] = pred
    	score[i]= mean_absolute_error(np.exp(y_test), pred)
    	print(score[i])
    	i+=1

    print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))


    print(mean_absolute_error(np.exp(label),submission.values))
    submission.to_csv(cache_path_output+"knn_retrain_%s.csv" %NumK, index_label='id')
