import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

retrain = pd.read_csv('../../GP/gptrainpredictions.csv', index_col=0)
pred_test = pd.read_csv('../../GP/gptestpredictions.csv', index_col=0)

mean_absolute_error(retrain['loss'], retrain['predictions1'])
mean_absolute_error(retrain['loss'], retrain['predictions2'])
mean_absolute_error(retrain['loss'], retrain['predictions3'])
mean_absolute_error(retrain['loss'], retrain['predictions4'])
mean_absolute_error(retrain['loss'], retrain['predictions5'])

train = pd.read_csv('../../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
MIN_VALUE = np.min(train['loss'])

def bound_df_retrain(df):
    assert df.shape[0] == 188318
    assert df.columns[0] == 'loss'
    df.loc[df['loss']>MAX_VALUE,:] = MAX_VALUE
    df.loc[df['loss']<MIN_VALUE,:] = MIN_VALUE

pred_gp_1_retrain = retrain['predictions1']
pred_gp_1_retrain = pd.DataFrame(pred_gp_1_retrain)
pred_gp_1_retrain.columns=['loss']
bound_df_retrain(pred_gp_1_retrain)

pred_gp_2_retrain = retrain['predictions2']
pred_gp_2_retrain = pd.DataFrame(pred_gp_2_retrain)
pred_gp_2_retrain.columns=['loss']
bound_df_retrain(pred_gp_2_retrain)

pred_gp_3_retrain = retrain['predictions3']
pred_gp_3_retrain = pd.DataFrame(pred_gp_3_retrain)
pred_gp_3_retrain.columns=['loss']
bound_df_retrain(pred_gp_3_retrain)

def bound_df_test(df):
    assert df.shape[0] == 125546
    assert df.columns[0] == 'loss'
    df.loc[df['loss']>MAX_VALUE,:] = MAX_VALUE
    df.loc[df['loss']<MIN_VALUE,:] = MIN_VALUE


pred_gp_1 = pred_test['predictions1']
pred_gp_1 = pd.DataFrame(pred_gp_1)
pred_gp_1.columns=['loss']
bound_df_test(pred_gp_1)

pred_gp_2 = pred_test['predictions2']
pred_gp_2 = pd.DataFrame(pred_gp_2)
pred_gp_2.columns=['loss']
bound_df_test(pred_gp_2)

pred_gp_3 = pred_test['predictions3']
pred_gp_3 = pd.DataFrame(pred_gp_3)
pred_gp_3.columns=['loss']
bound_df_test(pred_gp_3)




# ======================== optimize ======================== #
from scipy.optimize import minimize


args = [
    pred_gp_1_retrain['loss'].values,          #1
    pred_gp_2_retrain['loss'].values,          #2
    pred_gp_3_retrain['loss'].values,          #3
    train['loss'].values
]

print len(args)-1
pe= 7

def f(coord,args):
    #pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,pred_11,pred_12,pred_13,pred_14,pred_15,pred_16,pred_17,pred_18,r = args
    return np.mean( np.abs(coord[pe*0]*args[0] + coord[pe*0+1]*(args[0] ** 2) + coord[pe*0+2]*np.log(args[0]) + coord[pe*0+3]*1/(1.0+args[0]) + coord[pe*0+4]*(args[0] ** 0.5)
      + coord[pe*0+5]*np.sin(args[0]) + coord[pe*0+6]*np.cos(args[0])

      + coord[pe*1]*args[1] + coord[pe*1+1]*(args[1] ** 2) + coord[pe*1+2]*np.log(args[1]) + coord[pe*1+3]*1/(1.0+args[1]) + coord[pe*1+4]*(args[1] ** 0.5)
      + coord[pe*1+5]*np.sin(args[1]) + coord[pe*0+6]*np.cos(args[1])

      + coord[pe*2]*args[2] + coord[pe*1+1]*(args[2] ** 2) + coord[pe*2+2]*np.log(args[2]) + coord[pe*2+3]*1/(1.0+args[2]) + coord[pe*2+4]*(args[2] ** 0.5)
      + coord[pe*2+5]*np.sin(args[2]) + coord[pe*0+6]*np.cos(args[2])
     - args[-1])
     )


initial_guess = np.array([0.1 for x in range(pe * 3)])

Nfeval = 1
def callbackF(Xi):
    global Nfeval
    print '{0:4d}   {1: 3.6f}'.format(Nfeval, f(Xi, args))
    Nfeval += 1


res = minimize(f,initial_guess, args = args
                              ,method='SLSQP'
                              ,options={"maxiter":1000000,"disp":True}
                              ,callback=callbackF)

print res

gp_pred_retrain = (res.x[pe*0]*args[0] + res.x[pe*0+1]*(args[0] ** 2) + res.x[pe*0+2]*np.log(args[0]) + res.x[pe*0+3]*1/(1.0+args[0]) + res.x[pe*0+4]*(args[0] ** 0.5)
  + res.x[pe*0+5]*np.sin(args[0]) + res.x[pe*0+6]*np.cos(args[0])

  + res.x[pe*1]*args[1] + res.x[pe*1+1]*(args[1] ** 2) + res.x[pe*1+2]*np.log(args[1]) + res.x[pe*1+3]*1/(1.0+args[1]) + res.x[pe*1+4]*(args[1] ** 0.5)
  + res.x[pe*1+5]*np.sin(args[1]) + res.x[pe*0+6]*np.cos(args[1])

  + res.x[pe*2]*args[2] + res.x[pe*1+1]*(args[2] ** 2) + res.x[pe*2+2]*np.log(args[2]) + res.x[pe*2+3]*1/(1.0+args[2]) + res.x[pe*2+4]*(args[2] ** 0.5)
  + res.x[pe*2+5]*np.sin(args[2]) + res.x[pe*0+6]*np.cos(args[2]))

gp_pred_retrain = pd.DataFrame(gp_pred_retrain)
gp_pred_retrain.index = retrain.index
gp_pred_retrain.columns = ['loss']
print mean_absolute_error(train['loss'], gp_pred_retrain.values)
bound_df_retrain(gp_pred_retrain)

####### prediction
args = [
    pred_gp_1['loss'].values,          #1
    pred_gp_2['loss'].values,          #2
    pred_gp_3['loss'].values,          #3
]

gp_pred = (res.x[pe*0]*args[0] + res.x[pe*0+1]*(args[0] ** 2) + res.x[pe*0+2]*np.log(args[0]) + res.x[pe*0+3]*1/(1.0+args[0]) + res.x[pe*0+4]*(args[0] ** 0.5)
  + res.x[pe*0+5]*np.sin(args[0]) + res.x[pe*0+6]*np.cos(args[0])

  + res.x[pe*1]*args[1] + res.x[pe*1+1]*(args[1] ** 2) + res.x[pe*1+2]*np.log(args[1]) + res.x[pe*1+3]*1/(1.0+args[1]) + res.x[pe*1+4]*(args[1] ** 0.5)
  + res.x[pe*1+5]*np.sin(args[1]) + res.x[pe*0+6]*np.cos(args[1])

  + res.x[pe*2]*args[2] + res.x[pe*1+1]*(args[2] ** 2) + res.x[pe*2+2]*np.log(args[2]) + res.x[pe*2+3]*1/(1.0+args[2]) + res.x[pe*2+4]*(args[2] ** 0.5)
  + res.x[pe*2+5]*np.sin(args[2]) + res.x[pe*0+6]*np.cos(args[2]))

gp_pred = pd.DataFrame(gp_pred)
gp_pred.index = pred_test.index
gp_pred.columns = ['loss']
bound_df_test(gp_pred)



####### Load ver_42

ver_42_retrain = pd.read_csv('../ver_42/retrain.csv', index_col=0)
ver_42_retrain.index = retrain.index
ver_42 = pd.read_csv('../ver_42/pred_retrain.csv', index_col=0)
ver_42.index = pred_test.index








# ======================== optimize ======================== #
from scipy.optimize import minimize


args = [
    pred_gp_1_retrain['loss'].values,          #1
    ver_42_retrain['loss'].values,          #2
    train['loss'].values
]

print len(args)-1
pe= 7

def f(coord,args):
    #pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,pred_11,pred_12,pred_13,pred_14,pred_15,pred_16,pred_17,pred_18,r = args
    return np.mean( np.abs(coord[pe*0]*args[0] + coord[pe*0+1]*(args[0] ** 2) + coord[pe*0+2]*np.log(args[0]) + coord[pe*0+3]*1/(1.0+args[0]) + coord[pe*0+4]*(args[0] ** 0.5)
      + coord[pe*0+5]*np.sin(args[0]) + coord[pe*0+6]*np.cos(args[0])

      + coord[pe*1]*args[1] + coord[pe*1+1]*(args[1] ** 2) + coord[pe*1+2]*np.log(args[1]) + coord[pe*1+3]*1/(1.0+args[1]) + coord[pe*1+4]*(args[1] ** 0.5)
      + coord[pe*1+5]*np.sin(args[1]) + coord[pe*0+6]*np.cos(args[1])

     - args[-1])
     )


initial_guess = np.array([0.1 for x in range(pe * 3)])

Nfeval = 1
def callbackF(Xi):
    global Nfeval
    print '{0:4d}   {1: 3.6f}'.format(Nfeval, f(Xi, args))
    Nfeval += 1


res = minimize(f,initial_guess, args = args
                              ,method='SLSQP'
                              ,options={"maxiter":1000000,"disp":True}
                              ,callback=callbackF)

print res


####### prediction
args = [
    gp_pred['loss'].values,          #1
    ver_42['loss'].values,          #2
]

ensemble = (res.x[pe*0]*args[0] + res.x[pe*0+1]*(args[0] ** 2) + res.x[pe*0+2]*np.log(args[0]) + res.x[pe*0+3]*1/(1.0+args[0]) + res.x[pe*0+4]*(args[0] ** 0.5)
  + res.x[pe*0+5]*np.sin(args[0]) + res.x[pe*0+6]*np.cos(args[0])

  + res.x[pe*1]*args[1] + res.x[pe*1+1]*(args[1] ** 2) + res.x[pe*1+2]*np.log(args[1]) + res.x[pe*1+3]*1/(1.0+args[1]) + res.x[pe*1+4]*(args[1] ** 0.5)
  + res.x[pe*1+5]*np.sin(args[1]) + res.x[pe*0+6]*np.cos(args[1])
)

ensemble = pd.DataFrame(ensemble)
ensemble.index = ver_42.index
ensemble.columns = ['loss']
bound_df_test(ensemble)

ensemble.to_csv("pred_retrain.csv", index_label='id')
