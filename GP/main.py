import pandas as pd

retrain = pd.read_csv("gptrainpredictions.csv")
del retrain['loss']
retrain['loss'] = (retrain['predictions1']
                    + retrain['predictions2']
                    + retrain['predictions3']
                    + retrain['predictions4']
                    + retrain['predictions5']
                  )/5.
retrain = retrain.loc[:,["id","loss"]]
retrain.to_csv('gp_retrain.csv', index = False)


test = pd.read_csv("gptestpredictions.csv")
test['loss'] = (test['predictions1']
                    + test['predictions2']
                    + test['predictions3']
                    + test['predictions4']
                    + test['predictions5']
                  )/5.
test = test.loc[:,["id","loss"]]
retrain.to_csv('gp.csv', index = False)
