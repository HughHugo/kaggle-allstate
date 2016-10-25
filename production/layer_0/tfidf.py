import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

print("read training data")
train = pd.read_csv("../../input/train.csv")
target = train['loss']
NAME=train.columns[0:]
id_train =train['id']
del train['id']
del train['loss']

test  = pd.read_csv("../../input/test.csv")
id_test = test['id']
del test['id']

names_cat = ['cat' + str(i+1) for i in range(116)]
for i in names_cat:
    #print i
    le = LabelEncoder()
    le.fit(np.concatenate([train[i].values, test[i].values]))
    train[i] = le.transform(train[i])
    test[i] = le.transform(test[i])

transformer = TfidfTransformer()
train = transformer.fit_transform(train)

train=pd.DataFrame(train.toarray())
train.columns=NAME[1:-1]
train=pd.concat([id_train,train,target],axis=1)
train.to_csv("../../cache/layer_0/train_tfidf.csv",index=False)

test = transformer.transform(test)
test=pd.DataFrame(test.toarray())
test.columns=NAME[1:-1]
test=pd.concat([id_test,test],axis=1)
test.to_csv("../../cache/layer_0/test_tfidf.csv",index=False)
