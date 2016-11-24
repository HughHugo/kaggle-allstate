# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import gmean

train = pd.read_csv('../../input/train.csv', index_col=0)
MAX_VALUE = np.max(train['loss'])
MIN_VALUE = np.min(train['loss'])

def bound_df_retrain(df):
    assert df.shape[0] == 188318
    assert df.columns[0] == 'loss'
    df.loc[df['loss']>MAX_VALUE,:] = MAX_VALUE
    df.loc[df['loss']<MIN_VALUE,:] = MIN_VALUE


### Train ###
pred_nn_1_retrain = pd.read_csv('../../NN_1/NN_retrain_1.csv', index_col=0)
bound_df_retrain(pred_nn_1_retrain)
pred_nn_2_retrain = pd.read_csv('../../NN_2/NN_retrain_2.csv', index_col=0)
bound_df_retrain(pred_nn_2_retrain)
pred_nn_3_retrain = pd.read_csv('../../NN_3/NN_retrain_3.csv', index_col=0)
bound_df_retrain(pred_nn_3_retrain)
pred_nn_4_retrain = pd.read_csv('../../NN_4/NN_retrain_4.csv', index_col=0)
bound_df_retrain(pred_nn_4_retrain)
pred_nn_5_retrain = pd.read_csv('../../NN_5/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_nn_5_retrain)
pred_nn_6_retrain = pd.read_csv('../../NN_6/NN_retrain_6.csv', index_col=0)
bound_df_retrain(pred_nn_6_retrain)


pred_nn_1_fix_retrain = pd.read_csv('../../NN_1_fix/NN_retrain_1.csv', index_col=0)
bound_df_retrain(pred_nn_1_fix_retrain)
pred_nn_2_fix_retrain = pd.read_csv('../../NN_2_fix/NN_retrain_2.csv', index_col=0)
bound_df_retrain(pred_nn_2_fix_retrain)
pred_nn_3_fix_retrain = pd.read_csv('../../NN_3_fix/NN_retrain_3.csv', index_col=0)
bound_df_retrain(pred_nn_3_fix_retrain)
pred_nn_4_fix_retrain = pd.read_csv('../../NN_4_fix/NN_retrain_4.csv', index_col=0)
bound_df_retrain(pred_nn_4_fix_retrain)
pred_nn_5_fix_retrain = pd.read_csv('../../NN_5_fix/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_nn_5_fix_retrain)
pred_nn_6_fix_retrain = pd.read_csv('../../NN_6_fix/NN_retrain_6.csv', index_col=0)
bound_df_retrain(pred_nn_6_fix_retrain)


pred_new_nn_1_retrain = pd.read_csv('../../NEW_NN_1/NN_retrain_1.csv', index_col=0)
bound_df_retrain(pred_new_nn_1_retrain)
pred_new_nn_2_retrain = pd.read_csv('../../NEW_NN_2/NN_retrain_2.csv', index_col=0)
bound_df_retrain(pred_new_nn_2_retrain)
pred_new_nn_3_retrain = pd.read_csv('../../NEW_NN_3/NN_retrain_3.csv', index_col=0)
bound_df_retrain(pred_new_nn_3_retrain)
pred_new_nn_4_retrain = pd.read_csv('../../NEW_NN_4/NN_retrain_4.csv', index_col=0)
bound_df_retrain(pred_new_nn_4_retrain)
pred_new_nn_5_retrain = pd.read_csv('../../NEW_NN_5/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_retrain)
pred_new_nn_6_retrain = pd.read_csv('../../NEW_NN_6/NN_retrain_6.csv', index_col=0)
bound_df_retrain(pred_new_nn_6_retrain)
pred_new_nn_7_retrain = pd.read_csv('../../NEW_NN_7/NN_retrain_7.csv', index_col=0)
bound_df_retrain(pred_new_nn_7_retrain)


pred_new_nn_1_65_retrain = pd.read_csv('../../NEW_NN_1_65/NN_retrain_1.csv', index_col=0)
bound_df_retrain(pred_new_nn_1_65_retrain)
pred_new_nn_2_65_retrain = pd.read_csv('../../NEW_NN_2_65/NN_retrain_2.csv', index_col=0)
bound_df_retrain(pred_new_nn_2_65_retrain)


pred_xgb_1_retrain = pd.read_csv('../../XGB_1/XGB_retrain_1.csv', index_col=0)
bound_df_retrain(pred_xgb_1_retrain)
pred_xgb_2_retrain = pd.read_csv('../../XGB_2/XGB_retrain_2.csv', index_col=0)
bound_df_retrain(pred_xgb_2_retrain)
pred_xgb_3_retrain = pd.read_csv('../../XGB_3/XGB_retrain_3.csv', index_col=0)
bound_df_retrain(pred_xgb_3_retrain)
pred_xgb_6_retrain = pd.read_csv('../../XGB_6/XGB_retrain_6.csv', index_col=0)
bound_df_retrain(pred_xgb_6_retrain)
pred_xgb_9_retrain = pd.read_csv('../../XGB_9/XGB_retrain_9.csv', index_col=0)
bound_df_retrain(pred_xgb_9_retrain)
pred_xgb_10_retrain = pd.read_csv('../../XGB_10/XGB_retrain_10.csv', index_col=0)
bound_df_retrain(pred_xgb_10_retrain)
pred_xgb_11_retrain = pd.read_csv('../../XGB_11/XGB_retrain_11.csv', index_col=0)
bound_df_retrain(pred_xgb_11_retrain)
pred_xgb_13_retrain = pd.read_csv('../../XGB_13/XGB_retrain_13.csv', index_col=0)
bound_df_retrain(pred_xgb_13_retrain)
pred_xgb_14_retrain = pd.read_csv('../../XGB_14/XGB_retrain_14.csv', index_col=0)
bound_df_retrain(pred_xgb_14_retrain)
pred_xgb_15_retrain = pd.read_csv('../../XGB_15/XGB_retrain_15.csv', index_col=0)
bound_df_retrain(pred_xgb_15_retrain)
pred_xgb_17_retrain = pd.read_csv('../../XGB_17/XGB_retrain_17.csv', index_col=0)
bound_df_retrain(pred_xgb_17_retrain)
pred_xgb_17_2way_retrain = pd.read_csv('../../XGB_17_2way/XGB_retrain_17.csv', index_col=0)
bound_df_retrain(pred_xgb_17_2way_retrain)
pred_xgb_18_retrain = pd.read_csv('../../XGB_18/XGB_retrain_18.csv', index_col=0)
bound_df_retrain(pred_xgb_18_retrain)
pred_xgb_19_retrain = pd.read_csv('../../XGB_19/XGB_retrain_19.csv', index_col=0)
bound_df_retrain(pred_xgb_19_retrain)
pred_xgb_20_retrain = pd.read_csv('../../XGB_20/XGB_retrain_20.csv', index_col=0)
bound_df_retrain(pred_xgb_20_retrain)


def bound_df_test(df):
    assert df.shape[0] == 125546
    assert df.columns[0] == 'loss'
    df.loc[df['loss']>MAX_VALUE,:] = MAX_VALUE
    df.loc[df['loss']<MIN_VALUE,:] = MIN_VALUE

### Test ###
pred_nn_1 = pd.read_csv('../../NN_1/NN_1.csv', index_col=0)
bound_df_test(pred_nn_1)
pred_nn_2 = pd.read_csv('../../NN_2/NN_2.csv', index_col=0)
bound_df_test(pred_nn_2)
pred_nn_3 = pd.read_csv('../../NN_3/NN_3.csv', index_col=0)
bound_df_test(pred_nn_3)
pred_nn_4 = pd.read_csv('../../NN_4/NN_4.csv', index_col=0)
bound_df_test(pred_nn_4)
pred_nn_5 = pd.read_csv('../../NN_5/NN_5.csv', index_col=0)
bound_df_test(pred_nn_5)
pred_nn_6 = pd.read_csv('../../NN_6/NN_6.csv', index_col=0)
bound_df_test(pred_nn_6)


pred_nn_1_fix = pd.read_csv('../../NN_1_fix/NN_1.csv', index_col=0)
bound_df_test(pred_nn_1_fix)
pred_nn_2_fix = pd.read_csv('../../NN_2_fix/NN_2.csv', index_col=0)
bound_df_test(pred_nn_2_fix)
pred_nn_3_fix = pd.read_csv('../../NN_3_fix/NN_3.csv', index_col=0)
bound_df_test(pred_nn_3_fix)
pred_nn_4_fix = pd.read_csv('../../NN_4_fix/NN_4.csv', index_col=0)
bound_df_test(pred_nn_4_fix)
pred_nn_5_fix = pd.read_csv('../../NN_5_fix/NN_5.csv', index_col=0)
bound_df_test(pred_nn_5_fix)
pred_nn_6_fix = pd.read_csv('../../NN_6_fix/NN_6.csv', index_col=0)
bound_df_test(pred_nn_6_fix)


pred_new_nn_1 = pd.read_csv('../../NEW_NN_1/NN_1.csv', index_col=0)
bound_df_test(pred_new_nn_1)
pred_new_nn_2 = pd.read_csv('../../NEW_NN_2/NN_2.csv', index_col=0)
bound_df_test(pred_new_nn_2)
pred_new_nn_3 = pd.read_csv('../../NEW_NN_3/NN_3.csv', index_col=0)
bound_df_test(pred_new_nn_3)
pred_new_nn_4 = pd.read_csv('../../NEW_NN_4/NN_4.csv', index_col=0)
bound_df_test(pred_new_nn_4)
pred_new_nn_5 = pd.read_csv('../../NEW_NN_5/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5)
pred_new_nn_6 = pd.read_csv('../../NEW_NN_6/NN_6.csv', index_col=0)
bound_df_test(pred_new_nn_6)
pred_new_nn_7 = pd.read_csv('../../NEW_NN_7/NN_7.csv', index_col=0)
bound_df_test(pred_new_nn_7)


pred_new_nn_1_65 = pd.read_csv('../../NEW_NN_1_65/NN_1.csv', index_col=0)
bound_df_test(pred_new_nn_1_65)
pred_new_nn_2_65 = pd.read_csv('../../NEW_NN_2_65/NN_2.csv', index_col=0)
bound_df_test(pred_new_nn_2_65)


pred_xgb_1 = pd.read_csv('../../XGB_1/XGB_1.csv', index_col=0)
bound_df_test(pred_xgb_1)
pred_xgb_2 = pd.read_csv('../../XGB_2/XGB_2.csv', index_col=0)
bound_df_test(pred_xgb_2)
pred_xgb_3 = pd.read_csv('../../XGB_3/XGB_3.csv', index_col=0)
bound_df_test(pred_xgb_3)
pred_xgb_6 = pd.read_csv('../../XGB_6/XGB_6.csv', index_col=0)
bound_df_test(pred_xgb_6)
pred_xgb_9 = pd.read_csv('../../XGB_9/XGB_9.csv', index_col=0)
bound_df_test(pred_xgb_9)
pred_xgb_10 = pd.read_csv('../../XGB_10/XGB_10.csv', index_col=0)
bound_df_test(pred_xgb_10)
pred_xgb_11 = pd.read_csv('../../XGB_11/XGB_11.csv', index_col=0)
bound_df_test(pred_xgb_11)
pred_xgb_13 = pd.read_csv('../../XGB_13/XGB_13.csv', index_col=0)
bound_df_test(pred_xgb_13)
pred_xgb_14 = pd.read_csv('../../XGB_14/XGB_14.csv', index_col=0)
bound_df_test(pred_xgb_14)
pred_xgb_15 = pd.read_csv('../../XGB_15/XGB_15.csv', index_col=0)
bound_df_test(pred_xgb_15)
pred_xgb_17 = pd.read_csv('../../XGB_17/XGB_17.csv', index_col=0)
bound_df_test(pred_xgb_17)
pred_xgb_17_2way = pd.read_csv('../../XGB_17_2way/XGB_17.csv', index_col=0)
bound_df_test(pred_xgb_17_2way)
pred_xgb_18 = pd.read_csv('../../XGB_18/XGB_18.csv', index_col=0)
bound_df_test(pred_xgb_18)
pred_xgb_19 = pd.read_csv('../../XGB_19/XGB_19.csv', index_col=0)
bound_df_test(pred_xgb_19)
pred_xgb_20 = pd.read_csv('../../XGB_20/XGB_20.csv', index_col=0)
bound_df_test(pred_xgb_20)


print "#"
print mean_absolute_error(train['loss'], pred_nn_1_retrain)
print mean_absolute_error(train['loss'], pred_nn_2_retrain)
print mean_absolute_error(train['loss'], pred_nn_3_retrain)
print mean_absolute_error(train['loss'], pred_nn_4_retrain)
print mean_absolute_error(train['loss'], pred_nn_5_retrain)
print mean_absolute_error(train['loss'], pred_nn_6_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_nn_1_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_2_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_3_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_4_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_5_fix_retrain)
print mean_absolute_error(train['loss'], pred_nn_6_fix_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_new_nn_1_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_2_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_3_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_4_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_5_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_6_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_7_retrain)
print "#"
print mean_absolute_error(train['loss'], pred_new_nn_1_65_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_2_65_retrain)
print "#"
#
nn_retrain_pool = [pred_nn_1_retrain.values,
                pred_nn_2_retrain.values,
                pred_nn_3_retrain.values,
                pred_nn_4_retrain.values,
                pred_nn_5_retrain.values,
                pred_nn_6_retrain.values,
                pred_nn_1_fix_retrain.values,
                pred_nn_2_fix_retrain.values,
                pred_nn_3_fix_retrain.values,
                pred_nn_4_fix_retrain.values,
                pred_nn_5_fix_retrain.values,
                pred_nn_6_fix_retrain.values,
                pred_new_nn_1_retrain.values,
                pred_new_nn_2_retrain.values,
                pred_new_nn_3_retrain.values,
                pred_new_nn_4_retrain.values,
                pred_new_nn_5_retrain.values,
                pred_new_nn_6_retrain.values,
                pred_new_nn_7_retrain.values,
                pred_new_nn_1_65_retrain.values,
                pred_new_nn_2_65_retrain.values,
]

pred_nn_retrain = np.mean(nn_retrain_pool,axis=0)
pred_nn_retrain = np.array([x[0] for x in pred_nn_retrain])
print mean_absolute_error(train['loss'], pred_nn_retrain)
pred_nn_retrain = pd.DataFrame(pred_nn_retrain, columns=['loss'], index=pred_nn_1_retrain.index)

pred_nn_retrain_gmean = gmean(nn_retrain_pool,axis=0)
pred_nn_retrain_gmean = np.array([x[0] for x in pred_nn_retrain_gmean])
pred_nn_retrain_gmean = pd.DataFrame(pred_nn_retrain_gmean, columns=['loss'], index=pred_nn_1_retrain.index)

pred_nn_retrain_sd = np.log(np.std(nn_retrain_pool,axis=0))
pred_nn_retrain_sd = np.array([x[0] for x in pred_nn_retrain_sd])
pred_nn_retrain_sd = pd.DataFrame(pred_nn_retrain_sd, columns=['loss'], index=pred_nn_1_retrain.index)

pred_nn_retrain_mean_sd_1 = np.mean(nn_retrain_pool,axis=0) + np.std(nn_retrain_pool,axis=0)
pred_nn_retrain_mean_sd_1 = np.array([x[0] for x in pred_nn_retrain_mean_sd_1])
pred_nn_retrain_mean_sd_1 = pd.DataFrame(pred_nn_retrain_mean_sd_1, columns=['loss'], index=pred_nn_1_retrain.index)

pred_nn_retrain_mean_sd_2 = np.mean(nn_retrain_pool,axis=0) - np.std(nn_retrain_pool,axis=0)
pred_nn_retrain_mean_sd_2 = np.array([x[0] for x in pred_nn_retrain_mean_sd_2])
pred_nn_retrain_mean_sd_2 = pd.DataFrame(pred_nn_retrain_mean_sd_2, columns=['loss'], index=pred_nn_1_retrain.index)

pred_nn_retrain_mean_sd_3 = np.mean(nn_retrain_pool,axis=0) * np.std(nn_retrain_pool,axis=0)
pred_nn_retrain_mean_sd_3 = np.array([x[0] for x in pred_nn_retrain_mean_sd_3])
pred_nn_retrain_mean_sd_3 = pd.DataFrame(pred_nn_retrain_mean_sd_3, columns=['loss'], index=pred_nn_1_retrain.index)

pred_nn_retrain_mean_sd_4 = np.mean(nn_retrain_pool,axis=0) / np.std(nn_retrain_pool,axis=0)
pred_nn_retrain_mean_sd_4 = np.array([x[0] for x in pred_nn_retrain_mean_sd_4])
pred_nn_retrain_mean_sd_4 = pd.DataFrame(pred_nn_retrain_mean_sd_4, columns=['loss'], index=pred_nn_1_retrain.index)

print "##"
nn_pool = [pred_nn_1['loss'],
             pred_nn_2['loss'],
             pred_nn_3['loss'],
             pred_nn_4['loss'],
             pred_nn_5['loss'],
             pred_nn_6['loss'],
             pred_nn_1_fix['loss'],
             pred_nn_2_fix['loss'],
             pred_nn_3_fix['loss'],
             pred_nn_4_fix['loss'],
             pred_nn_5_fix['loss'],
             pred_nn_6_fix['loss'],
             pred_new_nn_1['loss'],
             pred_new_nn_2['loss'],
             pred_new_nn_3['loss'],
             pred_new_nn_4['loss'],
             pred_new_nn_5['loss'],
             pred_new_nn_6['loss'],
             pred_new_nn_7['loss'],
             pred_new_nn_1_65['loss'],
             pred_new_nn_2_65['loss'],
]
assert len(nn_pool) == len(nn_retrain_pool)


pred_nn = np.mean(nn_pool,axis=0)
pred_nn = pd.DataFrame(pred_nn, columns=['loss'], index=pred_nn_1.index)

pred_nn_gmean = gmean(nn_pool, axis=0)
pred_nn_gmean = pd.DataFrame(pred_nn_gmean, columns=['loss'], index=pred_nn_1.index)

pred_nn_sd = np.log(np.std(nn_pool, axis=0))
pred_nn_sd = pd.DataFrame(pred_nn_sd, columns=['loss'], index=pred_nn_1.index)

pred_nn_mean_sd_1 = np.mean(nn_pool,axis=0) + np.std(nn_pool, axis=0)
pred_nn_mean_sd_1 = pd.DataFrame(pred_nn_mean_sd_1, columns=['loss'], index=pred_nn_1.index)

pred_nn_mean_sd_2 = np.mean(nn_pool,axis=0) - np.std(nn_pool, axis=0)
pred_nn_mean_sd_2 = pd.DataFrame(pred_nn_mean_sd_2, columns=['loss'], index=pred_nn_1.index)

pred_nn_mean_sd_3 = np.mean(nn_pool,axis=0) * np.std(nn_pool, axis=0)
pred_nn_mean_sd_3 = pd.DataFrame(pred_nn_mean_sd_3, columns=['loss'], index=pred_nn_1.index)

pred_nn_mean_sd_4 = np.mean(nn_pool,axis=0) / np.std(nn_pool, axis=0)
pred_nn_mean_sd_4 = pd.DataFrame(pred_nn_mean_sd_4, columns=['loss'], index=pred_nn_1.index)

##############################################################################################################
print mean_absolute_error(train['loss'], pred_xgb_1_retrain)
print mean_absolute_error(train['loss'], pred_xgb_2_retrain)
print mean_absolute_error(train['loss'], pred_xgb_3_retrain)
print mean_absolute_error(train['loss'], pred_xgb_6_retrain)
print mean_absolute_error(train['loss'], pred_xgb_9_retrain)
print mean_absolute_error(train['loss'], pred_xgb_10_retrain)
print mean_absolute_error(train['loss'], pred_xgb_11_retrain)
print mean_absolute_error(train['loss'], pred_xgb_13_retrain)
print mean_absolute_error(train['loss'], pred_xgb_14_retrain)
print mean_absolute_error(train['loss'], pred_xgb_15_retrain)
print mean_absolute_error(train['loss'], pred_xgb_17_retrain)
print mean_absolute_error(train['loss'], pred_xgb_17_2way_retrain)
print mean_absolute_error(train['loss'], pred_xgb_18_retrain)
print mean_absolute_error(train['loss'], pred_xgb_19_retrain)
print mean_absolute_error(train['loss'], pred_xgb_20_retrain)

xgb_retrain_pool = [pred_xgb_1_retrain.values,
                pred_xgb_2_retrain.values,
                pred_xgb_3_retrain.values,
                pred_xgb_6_retrain.values,
                pred_xgb_9_retrain.values,
                pred_xgb_10_retrain.values,
                pred_xgb_11_retrain.values,
                pred_xgb_13_retrain.values,
                pred_xgb_14_retrain.values,
                pred_xgb_15_retrain.values,
                pred_xgb_17_retrain.values,
                pred_xgb_17_2way_retrain.values,
                pred_xgb_18_retrain.values,
                pred_xgb_19_retrain.values,
                pred_xgb_20_retrain.values
                ]

pred_xgb_retrain = np.mean(xgb_retrain_pool,axis=0)
pred_xgb_retrain = np.array([x[0] for x in pred_xgb_retrain])
print mean_absolute_error(train['loss'], pred_xgb_retrain)
pred_xgb_retrain = pd.DataFrame(pred_xgb_retrain, columns=['loss'], index=pred_xgb_1_retrain.index)
print "##"


pred_xgb_retrain_gmean = gmean(xgb_retrain_pool,axis=0)
pred_xgb_retrain_gmean = np.array([x[0] for x in pred_xgb_retrain_gmean])
pred_xgb_retrain_gmean = pd.DataFrame(pred_xgb_retrain_gmean, columns=['loss'], index=pred_xgb_1_retrain.index)

pred_xgb_retrain_sd = np.log(np.std(xgb_retrain_pool,axis=0))
pred_xgb_retrain_sd = np.array([x[0] for x in pred_xgb_retrain_sd])
pred_xgb_retrain_sd = pd.DataFrame(pred_xgb_retrain_sd, columns=['loss'], index=pred_xgb_1_retrain.index)

pred_xgb_retrain_mean_sd_1 = np.mean(xgb_retrain_pool,axis=0) + np.std(xgb_retrain_pool,axis=0)
pred_xgb_retrain_mean_sd_1 = np.array([x[0] for x in pred_xgb_retrain_mean_sd_1])
pred_xgb_retrain_mean_sd_1 = pd.DataFrame(pred_xgb_retrain_mean_sd_1, columns=['loss'], index=pred_xgb_1_retrain.index)

pred_xgb_retrain_mean_sd_2 = np.mean(xgb_retrain_pool,axis=0) - np.std(xgb_retrain_pool,axis=0)
pred_xgb_retrain_mean_sd_2 = np.array([x[0] for x in pred_xgb_retrain_mean_sd_2])
pred_xgb_retrain_mean_sd_2 = pd.DataFrame(pred_xgb_retrain_mean_sd_2, columns=['loss'], index=pred_xgb_1_retrain.index)

pred_xgb_retrain_mean_sd_3 = np.mean(xgb_retrain_pool,axis=0) * np.std(xgb_retrain_pool,axis=0)
pred_xgb_retrain_mean_sd_3 = np.array([x[0] for x in pred_xgb_retrain_mean_sd_3])
pred_xgb_retrain_mean_sd_3 = pd.DataFrame(pred_xgb_retrain_mean_sd_3, columns=['loss'], index=pred_xgb_1_retrain.index)

pred_xgb_retrain_mean_sd_4 = np.mean(xgb_retrain_pool,axis=0) / np.std(xgb_retrain_pool,axis=0)
pred_xgb_retrain_mean_sd_4 = np.array([x[0] for x in pred_xgb_retrain_mean_sd_4])
pred_xgb_retrain_mean_sd_4 = pd.DataFrame(pred_xgb_retrain_mean_sd_4, columns=['loss'], index=pred_xgb_1_retrain.index)


xgb_pool = [pred_xgb_1['loss'],
              pred_xgb_2['loss'],
              pred_xgb_3['loss'],
              pred_xgb_6['loss'],
              pred_xgb_9['loss'],
              pred_xgb_10['loss'],
              pred_xgb_11['loss'],
              pred_xgb_13['loss'],
              pred_xgb_14['loss'],
              pred_xgb_15['loss'],
              pred_xgb_17['loss'],
              pred_xgb_17_2way['loss'],
              pred_xgb_18['loss'],
              pred_xgb_19['loss'],
              pred_xgb_20['loss'],
        ]
assert len(xgb_pool) == len(xgb_retrain_pool)

pred_xgb = np.mean(xgb_pool, axis=0)
pred_xgb = pd.DataFrame(pred_xgb, columns=['loss'], index=pred_xgb_1.index)

pred_xgb_gmean = gmean(xgb_pool, axis=0)
pred_xgb_gmean = pd.DataFrame(pred_xgb_gmean, columns=['loss'], index=pred_xgb_1.index)

pred_xgb_sd = np.log(np.std(xgb_pool, axis=0))
pred_xgb_sd = pd.DataFrame(pred_xgb_sd, columns=['loss'], index=pred_xgb_1.index)

pred_xgb_mean_sd_1 = np.mean(xgb_pool,axis=0) + np.std(xgb_pool, axis=0)
pred_xgb_mean_sd_1 = pd.DataFrame(pred_xgb_mean_sd_1, columns=['loss'], index=pred_xgb_1.index)

pred_xgb_mean_sd_2 = np.mean(xgb_pool,axis=0) - np.std(xgb_pool, axis=0)
pred_xgb_mean_sd_2 = pd.DataFrame(pred_xgb_mean_sd_2, columns=['loss'], index=pred_xgb_1.index)

pred_xgb_mean_sd_3 = np.mean(xgb_pool,axis=0) * np.std(xgb_pool, axis=0)
pred_xgb_mean_sd_3 = pd.DataFrame(pred_xgb_mean_sd_3, columns=['loss'], index=pred_xgb_1.index)

pred_xgb_mean_sd_4 = np.mean(xgb_pool,axis=0) / np.std(xgb_pool, axis=0)
pred_xgb_mean_sd_4 = pd.DataFrame(pred_xgb_mean_sd_4, columns=['loss'], index=pred_xgb_1.index)


# ======================== optimize ======================== #
from scipy.optimize import minimize


args = [
    pred_nn_1_retrain['loss'].values,           #1
    pred_nn_2_retrain['loss'].values,           #2
    pred_nn_3_retrain['loss'].values,           #3
    pred_nn_4_retrain['loss'].values,           #4
    pred_nn_5_retrain['loss'].values,           #5
    pred_nn_6_retrain['loss'].values,           #6
    pred_nn_1_fix_retrain['loss'].values,       #7
    pred_nn_2_fix_retrain['loss'].values,       #8
    pred_nn_3_fix_retrain['loss'].values,       #9
    pred_nn_4_fix_retrain['loss'].values,       #10
    pred_nn_5_fix_retrain['loss'].values,       #11
    pred_nn_6_fix_retrain['loss'].values,       #12
    pred_new_nn_1_retrain['loss'].values,       #13
    pred_new_nn_2_retrain['loss'].values,       #14
    pred_new_nn_3_retrain['loss'].values,       #15
    pred_new_nn_4_retrain['loss'].values,       #16
    pred_new_nn_5_retrain['loss'].values,       #17
    pred_new_nn_6_retrain['loss'].values,       #18
    pred_new_nn_7_retrain['loss'].values,       #19
    pred_new_nn_1_65_retrain['loss'].values,    #20
    pred_new_nn_2_65_retrain['loss'].values,    #21
    pred_xgb_1_retrain['loss'].values,          #22
    pred_xgb_2_retrain['loss'].values,          #23
    pred_xgb_3_retrain['loss'].values,          #24
    pred_xgb_6_retrain['loss'].values,          #25
    pred_xgb_9_retrain['loss'].values,          #26
    pred_xgb_10_retrain['loss'].values,         #27
    pred_xgb_11_retrain['loss'].values,         #28
    pred_xgb_13_retrain['loss'].values,         #29
    pred_xgb_14_retrain['loss'].values,         #30
    pred_xgb_15_retrain['loss'].values,         #31
    pred_xgb_17_retrain['loss'].values,         #32
    pred_xgb_17_2way_retrain['loss'].values,    #33
    pred_xgb_18_retrain['loss'].values,         #34
    pred_xgb_19_retrain['loss'].values,         #35
    pred_xgb_20_retrain['loss'].values,         #36 ##########
    pred_nn_retrain['loss'].values,             #37
    pred_xgb_retrain['loss'].values,            #38
    pred_nn_retrain_gmean['loss'].values,       #39
    pred_xgb_retrain_gmean['loss'].values,      #40
    pred_nn_retrain_sd['loss'].values,          #41
    pred_xgb_retrain_sd['loss'].values,         #42
    pred_nn_retrain_mean_sd_1['loss'].values,   #43
    pred_xgb_retrain_mean_sd_1['loss'].values,  #44
    pred_nn_retrain_mean_sd_2['loss'].values,   #45
    pred_xgb_retrain_mean_sd_2['loss'].values,  #46
    pred_nn_retrain_mean_sd_3['loss'].values,   #47
    pred_xgb_retrain_mean_sd_3['loss'].values,  #48
    pred_nn_retrain_mean_sd_4['loss'].values,   #49
    pred_xgb_retrain_mean_sd_4['loss'].values,  #50
    train['loss'].values
]

print len(args)-1
pe= 5

def f(coord,args):
    #pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,pred_11,pred_12,pred_13,pred_14,pred_15,pred_16,pred_17,pred_18,r = args
    return np.mean( np.abs(coord[pe*0]*args[0] + coord[pe*0+1]*(args[0] ** 2) + coord[pe*0+2]*np.log(args[0]) + coord[pe*0+3]*1/(1.0+args[0]) + coord[pe*0+4]*(args[0] ** 0.5)
     + coord[pe*1]*args[1] + coord[pe*1+1]*(args[1] ** 2) + coord[pe*1+2]*np.log(args[1]) + coord[pe*1+3]*1/(1.0+args[1]) + coord[pe*1+4]*(args[1] ** 0.5)
     + coord[pe*2]*args[2] + coord[pe*2+1]*(args[2] ** 2) + coord[pe*2+2]*np.log(args[2]) + coord[pe*2+3]*1/(1.0+args[2]) + coord[pe*2+4]*(args[2] ** 0.5)
     + coord[pe*3]*args[3] + coord[pe*3+1]*(args[3] ** 2) + coord[pe*3+2]*np.log(args[3]) + coord[pe*3+3]*1/(1.0+args[3]) + coord[pe*3+4]*(args[3] ** 0.5)
     + coord[pe*4]*args[4] + coord[pe*4+1]*(args[4] ** 2) + coord[pe*4+2]*np.log(args[4]) + coord[pe*4+3]*1/(1.0+args[4]) + coord[pe*4+4]*(args[4] ** 0.5)
     + coord[pe*5]*args[5] + coord[pe*5+1]*(args[5] ** 2) + coord[pe*5+2]*np.log(args[5]) + coord[pe*5+3]*1/(1.0+args[5]) + coord[pe*5+4]*(args[5] ** 0.5)
     + coord[pe*6]*args[6] + coord[pe*6+1]*(args[6] ** 2) + coord[pe*6+2]*np.log(args[6]) + coord[pe*6+3]*1/(1.0+args[6]) + coord[pe*6+4]*(args[6] ** 0.5)
     + coord[pe*7]*args[7] + coord[pe*7+1]*(args[7] ** 2) + coord[pe*7+2]*np.log(args[7]) + coord[pe*7+3]*1/(1.0+args[7]) + coord[pe*7+4]*(args[7] ** 0.5)
     + coord[pe*8]*args[8] + coord[pe*8+1]*(args[8] ** 2) + coord[pe*8+2]*np.log(args[8]) + coord[pe*8+3]*1/(1.0+args[8]) + coord[pe*8+4]*(args[8] ** 0.5)
     + coord[pe*9]*args[9] + coord[pe*9+1]*(args[9] ** 2) + coord[pe*9+2]*np.log(args[9]) + coord[pe*9+3]*1/(1.0+args[9]) + coord[pe*9+4]*(args[9] ** 0.5)
     + coord[pe*10]*args[10] + coord[pe*10+1]*(args[10] ** 2) + coord[pe*10+2]*np.log(args[10]) + coord[pe*10+3]*1/(1.0+args[10]) + coord[pe*10+4]*(args[10] ** 0.5)
     + coord[pe*11]*args[11] + coord[pe*11+1]*(args[11] ** 2) + coord[pe*11+2]*np.log(args[11]) + coord[pe*11+3]*1/(1.0+args[11]) + coord[pe*11+4]*(args[11] ** 0.5)
     + coord[pe*12]*args[12] + coord[pe*12+1]*(args[12] ** 2) + coord[pe*12+2]*np.log(args[12]) + coord[pe*12+3]*1/(1.0+args[12]) + coord[pe*12+4]*(args[12] ** 0.5)
     + coord[pe*13]*args[13] + coord[pe*13+1]*(args[13] ** 2) + coord[pe*13+2]*np.log(args[13]) + coord[pe*13+3]*1/(1.0+args[13]) + coord[pe*13+4]*(args[13] ** 0.5)
     + coord[pe*14]*args[14] + coord[pe*14+1]*(args[14] ** 2) + coord[pe*14+2]*np.log(args[14]) + coord[pe*14+3]*1/(1.0+args[14]) + coord[pe*14+4]*(args[14] ** 0.5)
     + coord[pe*15]*args[15] + coord[pe*15+1]*(args[15] ** 2) + coord[pe*15+2]*np.log(args[15]) + coord[pe*15+3]*1/(1.0+args[15]) + coord[pe*15+4]*(args[15] ** 0.5)
     + coord[pe*16]*args[16] + coord[pe*16+1]*(args[16] ** 2) + coord[pe*16+2]*np.log(args[16]) + coord[pe*16+3]*1/(1.0+args[16]) + coord[pe*16+4]*(args[16] ** 0.5)
     + coord[pe*17]*args[17] + coord[pe*17+1]*(args[17] ** 2) + coord[pe*17+2]*np.log(args[17]) + coord[pe*17+3]*1/(1.0+args[17]) + coord[pe*17+4]*(args[17] ** 0.5)
     + coord[pe*18]*args[18] + coord[pe*18+1]*(args[18] ** 2) + coord[pe*18+2]*np.log(args[18]) + coord[pe*18+3]*1/(1.0+args[18]) + coord[pe*18+4]*(args[18] ** 0.5)
     + coord[pe*19]*args[19] + coord[pe*19+1]*(args[19] ** 2) + coord[pe*19+2]*np.log(args[19]) + coord[pe*19+3]*1/(1.0+args[19]) + coord[pe*19+4]*(args[19] ** 0.5)
     + coord[pe*20]*args[20] + coord[pe*20+1]*(args[20] ** 2) + coord[pe*20+2]*np.log(args[20]) + coord[pe*20+3]*1/(1.0+args[20]) + coord[pe*20+4]*(args[20] ** 0.5)
     + coord[pe*21]*args[21] + coord[pe*21+1]*(args[21] ** 2) + coord[pe*21+2]*np.log(args[21]) + coord[pe*21+3]*1/(1.0+args[21]) + coord[pe*21+4]*(args[21] ** 0.5)
     + coord[pe*22]*args[22] + coord[pe*22+1]*(args[22] ** 2) + coord[pe*22+2]*np.log(args[22]) + coord[pe*22+3]*1/(1.0+args[22]) + coord[pe*22+4]*(args[22] ** 0.5)
     + coord[pe*23]*args[23] + coord[pe*23+1]*(args[23] ** 2) + coord[pe*23+2]*np.log(args[23]) + coord[pe*23+3]*1/(1.0+args[23]) + coord[pe*23+4]*(args[23] ** 0.5)
     + coord[pe*24]*args[24] + coord[pe*24+1]*(args[24] ** 2) + coord[pe*24+2]*np.log(args[24]) + coord[pe*24+3]*1/(1.0+args[24]) + coord[pe*24+4]*(args[24] ** 0.5)
     + coord[pe*25]*args[25] + coord[pe*25+1]*(args[25] ** 2) + coord[pe*25+2]*np.log(args[25]) + coord[pe*25+3]*1/(1.0+args[25]) + coord[pe*25+4]*(args[25] ** 0.5)
     + coord[pe*26]*args[26] + coord[pe*26+1]*(args[26] ** 2) + coord[pe*26+2]*np.log(args[26]) + coord[pe*26+3]*1/(1.0+args[26]) + coord[pe*26+4]*(args[26] ** 0.5)
     + coord[pe*27]*args[27] + coord[pe*27+1]*(args[27] ** 2) + coord[pe*27+2]*np.log(args[27]) + coord[pe*27+3]*1/(1.0+args[27]) + coord[pe*27+4]*(args[27] ** 0.5)
     + coord[pe*28]*args[28] + coord[pe*28+1]*(args[28] ** 2) + coord[pe*28+2]*np.log(args[28]) + coord[pe*28+3]*1/(1.0+args[28]) + coord[pe*28+4]*(args[28] ** 0.5)
     + coord[pe*29]*args[29] + coord[pe*29+1]*(args[29] ** 2) + coord[pe*29+2]*np.log(args[29]) + coord[pe*29+3]*1/(1.0+args[29]) + coord[pe*29+4]*(args[29] ** 0.5)
     + coord[pe*30]*args[30] + coord[pe*30+1]*(args[30] ** 2) + coord[pe*30+2]*np.log(args[30]) + coord[pe*30+3]*1/(1.0+args[30]) + coord[pe*30+4]*(args[30] ** 0.5)
     + coord[pe*31]*args[31] + coord[pe*31+1]*(args[31] ** 2) + coord[pe*31+2]*np.log(args[31]) + coord[pe*31+3]*1/(1.0+args[31]) + coord[pe*31+4]*(args[31] ** 0.5)
     + coord[pe*32]*args[32] + coord[pe*32+1]*(args[32] ** 2) + coord[pe*32+2]*np.log(args[32]) + coord[pe*32+3]*1/(1.0+args[32]) + coord[pe*32+4]*(args[32] ** 0.5)
     + coord[pe*33]*args[33] + coord[pe*33+1]*(args[33] ** 2) + coord[pe*33+2]*np.log(args[33]) + coord[pe*33+3]*1/(1.0+args[33]) + coord[pe*33+4]*(args[33] ** 0.5)
     + coord[pe*34]*args[34] + coord[pe*34+1]*(args[34] ** 2) + coord[pe*34+2]*np.log(args[34]) + coord[pe*34+3]*1/(1.0+args[34]) + coord[pe*34+4]*(args[34] ** 0.5)
     + coord[pe*35]*args[35] + coord[pe*35+1]*(args[35] ** 2) + coord[pe*35+2]*np.log(args[35]) + coord[pe*35+3]*1/(1.0+args[35]) + coord[pe*35+4]*(args[35] ** 0.5)
     + coord[pe*36+0]*( args[36] - args[37] )
     + coord[pe*36+1]*( (args[36] - args[37]) ** 2 )
     + coord[pe*36+2]*( np.log(abs(args[36] - args[37])) )
     + coord[pe*36+3]*( 1/(1.0+args[36]-args[37]) )
     + coord[pe*36+4]*( abs(args[36]-args[37]) ** 0.5 )
     + coord[pe*36+5]*( np.sin(args[36]-args[37]) )
     + coord[pe*36+6]*( np.cos(args[36]-args[37]) )
     + coord[pe*36+7]*( args[38] - args[39] )
     + coord[pe*36+8]*( (args[38] - args[39]) ** 2 )
     + coord[pe*36+9]*( np.log(abs(args[38] - args[39])) )
     + coord[pe*36+10]*( 1/(1.0+args[38]-args[39]) )
     + coord[pe*36+11]*( abs(args[38]-args[39]) ** 0.5 )
     + coord[pe*36+12]*( np.sin(args[38]-args[39]) )
     + coord[pe*36+13]*( np.cos(args[38]-args[39]) )
     + coord[pe*36+14]*( args[40] - args[41] )
     + coord[pe*36+15]*( (args[40] - args[41]) ** 2 )
     + coord[pe*36+16]*( np.log(abs(args[40] - args[41])) )
     + coord[pe*36+17]*( 1/(1.0+args[40]-args[41]) )
     + coord[pe*36+18]*( abs(args[40]-args[41]) ** 0.5 )
     + coord[pe*36+19]*( np.sin(args[40]-args[41]) )
     + coord[pe*36+20]*( np.cos(args[40]-args[41]) )
     + coord[pe*36+21]*( args[42] - args[43] )
     + coord[pe*36+22]*( (args[42] - args[43]) ** 2 )
     + coord[pe*36+23]*( np.log(abs(args[42] - args[43])) )
     + coord[pe*36+24]*( 1/(1.0+args[42]-args[43]) )
     + coord[pe*36+25]*( abs(args[42]-args[43]) ** 0.5 )
     + coord[pe*36+26]*( np.sin(args[42]-args[43]) )
     + coord[pe*36+27]*( np.cos(args[42]-args[43]) )
     + coord[pe*36+28]*( args[44] - args[45] )
     + coord[pe*36+29]*( (args[44] - args[45]) ** 2 )
     + coord[pe*36+30]*( np.log(abs(args[44] - args[45])) )
     + coord[pe*36+31]*( 1/(1.0+args[44]-args[45]) )
     + coord[pe*36+32]*( abs(args[44]-args[45]) ** 0.5 )
     + coord[pe*36+33]*( np.sin(args[44]-args[45]) )
     + coord[pe*36+34]*( np.cos(args[44]-args[45]) )
     + coord[pe*36+35]*( args[46] - args[47] )
     + coord[pe*36+36]*( (args[46] - args[47]) ** 2 )
     + coord[pe*36+37]*( np.log(abs(args[46] - args[47])) )
     + coord[pe*36+38]*( 1/(1.0+args[46]-args[47]) )
     + coord[pe*36+39]*( abs(args[46]-args[47]) ** 0.5 )
     + coord[pe*36+40]*( np.sin(args[46]-args[47]) )
     + coord[pe*36+41]*( np.cos(args[46]-args[47]) )
     + coord[pe*36+42]*( args[48] - args[49] )
     + coord[pe*36+43]*( (args[48] - args[49]) ** 2 )
     + coord[pe*36+44]*( np.log(abs(args[48] - args[49])) )
     + coord[pe*36+45]*( 1/(1.0+args[48]-args[49]) )
     + coord[pe*36+46]*( abs(args[48]-args[49]) ** 0.5 )
     + coord[pe*36+47]*( np.sin(args[48]-args[49]) )
     + coord[pe*36+48]*( np.cos(args[48]-args[49]) )
     - args[-1]) )


initial_guess = np.array([0.00001 for x in range(pe * 36 + 49)])

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

pred_ensemble = (res.x[pe*0]*args[0] + res.x[pe*0+1]*(args[0] ** 2) + res.x[pe*0+2]*np.log(args[0]) + res.x[pe*0+3]*1/(1.0+args[0]) + res.x[pe*0+4]*(args[0] ** 0.5)
 + res.x[pe*1]*args[1] + res.x[pe*1+1]*(args[1] ** 2) + res.x[pe*1+2]*np.log(args[1]) + res.x[pe*1+3]*1/(1.0+args[1]) + res.x[pe*1+4]*(args[1] ** 0.5)
 + res.x[pe*2]*args[2] + res.x[pe*2+1]*(args[2] ** 2) + res.x[pe*2+2]*np.log(args[2]) + res.x[pe*2+3]*1/(1.0+args[2]) + res.x[pe*2+4]*(args[2] ** 0.5)
 + res.x[pe*3]*args[3] + res.x[pe*3+1]*(args[3] ** 2) + res.x[pe*3+2]*np.log(args[3]) + res.x[pe*3+3]*1/(1.0+args[3]) + res.x[pe*3+4]*(args[3] ** 0.5)
 + res.x[pe*4]*args[4] + res.x[pe*4+1]*(args[4] ** 2) + res.x[pe*4+2]*np.log(args[4]) + res.x[pe*4+3]*1/(1.0+args[4]) + res.x[pe*4+4]*(args[4] ** 0.5)
 + res.x[pe*5]*args[5] + res.x[pe*5+1]*(args[5] ** 2) + res.x[pe*5+2]*np.log(args[5]) + res.x[pe*5+3]*1/(1.0+args[5]) + res.x[pe*5+4]*(args[5] ** 0.5)
 + res.x[pe*6]*args[6] + res.x[pe*6+1]*(args[6] ** 2) + res.x[pe*6+2]*np.log(args[6]) + res.x[pe*6+3]*1/(1.0+args[6]) + res.x[pe*6+4]*(args[6] ** 0.5)
 + res.x[pe*7]*args[7] + res.x[pe*7+1]*(args[7] ** 2) + res.x[pe*7+2]*np.log(args[7]) + res.x[pe*7+3]*1/(1.0+args[7]) + res.x[pe*7+4]*(args[7] ** 0.5)
 + res.x[pe*8]*args[8] + res.x[pe*8+1]*(args[8] ** 2) + res.x[pe*8+2]*np.log(args[8]) + res.x[pe*8+3]*1/(1.0+args[8]) + res.x[pe*8+4]*(args[8] ** 0.5)
 + res.x[pe*9]*args[9] + res.x[pe*9+1]*(args[9] ** 2) + res.x[pe*9+2]*np.log(args[9]) + res.x[pe*9+3]*1/(1.0+args[9]) + res.x[pe*9+4]*(args[9] ** 0.5)
 + res.x[pe*10]*args[10] + res.x[pe*10+1]*(args[10] ** 2) + res.x[pe*10+2]*np.log(args[10]) + res.x[pe*10+3]*1/(1.0+args[10]) + res.x[pe*10+4]*(args[10] ** 0.5)
 + res.x[pe*11]*args[11] + res.x[pe*11+1]*(args[11] ** 2) + res.x[pe*11+2]*np.log(args[11]) + res.x[pe*11+3]*1/(1.0+args[11]) + res.x[pe*11+4]*(args[11] ** 0.5)
 + res.x[pe*12]*args[12] + res.x[pe*12+1]*(args[12] ** 2) + res.x[pe*12+2]*np.log(args[12]) + res.x[pe*12+3]*1/(1.0+args[12]) + res.x[pe*12+4]*(args[12] ** 0.5)
 + res.x[pe*13]*args[13] + res.x[pe*13+1]*(args[13] ** 2) + res.x[pe*13+2]*np.log(args[13]) + res.x[pe*13+3]*1/(1.0+args[13]) + res.x[pe*13+4]*(args[13] ** 0.5)
 + res.x[pe*14]*args[14] + res.x[pe*14+1]*(args[14] ** 2) + res.x[pe*14+2]*np.log(args[14]) + res.x[pe*14+3]*1/(1.0+args[14]) + res.x[pe*14+4]*(args[14] ** 0.5)
 + res.x[pe*15]*args[15] + res.x[pe*15+1]*(args[15] ** 2) + res.x[pe*15+2]*np.log(args[15]) + res.x[pe*15+3]*1/(1.0+args[15]) + res.x[pe*15+4]*(args[15] ** 0.5)
 + res.x[pe*16]*args[16] + res.x[pe*16+1]*(args[16] ** 2) + res.x[pe*16+2]*np.log(args[16]) + res.x[pe*16+3]*1/(1.0+args[16]) + res.x[pe*16+4]*(args[16] ** 0.5)
 + res.x[pe*17]*args[17] + res.x[pe*17+1]*(args[17] ** 2) + res.x[pe*17+2]*np.log(args[17]) + res.x[pe*17+3]*1/(1.0+args[17]) + res.x[pe*17+4]*(args[17] ** 0.5)
 + res.x[pe*18]*args[18] + res.x[pe*18+1]*(args[18] ** 2) + res.x[pe*18+2]*np.log(args[18]) + res.x[pe*18+3]*1/(1.0+args[18]) + res.x[pe*18+4]*(args[18] ** 0.5)
 + res.x[pe*19]*args[19] + res.x[pe*19+1]*(args[19] ** 2) + res.x[pe*19+2]*np.log(args[19]) + res.x[pe*19+3]*1/(1.0+args[19]) + res.x[pe*19+4]*(args[19] ** 0.5)
 + res.x[pe*20]*args[20] + res.x[pe*20+1]*(args[20] ** 2) + res.x[pe*20+2]*np.log(args[20]) + res.x[pe*20+3]*1/(1.0+args[20]) + res.x[pe*20+4]*(args[20] ** 0.5)
 + res.x[pe*21]*args[21] + res.x[pe*21+1]*(args[21] ** 2) + res.x[pe*21+2]*np.log(args[21]) + res.x[pe*21+3]*1/(1.0+args[21]) + res.x[pe*21+4]*(args[21] ** 0.5)
 + res.x[pe*22]*args[22] + res.x[pe*22+1]*(args[22] ** 2) + res.x[pe*22+2]*np.log(args[22]) + res.x[pe*22+3]*1/(1.0+args[22]) + res.x[pe*22+4]*(args[22] ** 0.5)
 + res.x[pe*23]*args[23] + res.x[pe*23+1]*(args[23] ** 2) + res.x[pe*23+2]*np.log(args[23]) + res.x[pe*23+3]*1/(1.0+args[23]) + res.x[pe*23+4]*(args[23] ** 0.5)
 + res.x[pe*24]*args[24] + res.x[pe*24+1]*(args[24] ** 2) + res.x[pe*24+2]*np.log(args[24]) + res.x[pe*24+3]*1/(1.0+args[24]) + res.x[pe*24+4]*(args[24] ** 0.5)
 + res.x[pe*25]*args[25] + res.x[pe*25+1]*(args[25] ** 2) + res.x[pe*25+2]*np.log(args[25]) + res.x[pe*25+3]*1/(1.0+args[25]) + res.x[pe*25+4]*(args[25] ** 0.5)
 + res.x[pe*26]*args[26] + res.x[pe*26+1]*(args[26] ** 2) + res.x[pe*26+2]*np.log(args[26]) + res.x[pe*26+3]*1/(1.0+args[26]) + res.x[pe*26+4]*(args[26] ** 0.5)
 + res.x[pe*27]*args[27] + res.x[pe*27+1]*(args[27] ** 2) + res.x[pe*27+2]*np.log(args[27]) + res.x[pe*27+3]*1/(1.0+args[27]) + res.x[pe*27+4]*(args[27] ** 0.5)
 + res.x[pe*28]*args[28] + res.x[pe*28+1]*(args[28] ** 2) + res.x[pe*28+2]*np.log(args[28]) + res.x[pe*28+3]*1/(1.0+args[28]) + res.x[pe*28+4]*(args[28] ** 0.5)
 + res.x[pe*29]*args[29] + res.x[pe*29+1]*(args[29] ** 2) + res.x[pe*29+2]*np.log(args[29]) + res.x[pe*29+3]*1/(1.0+args[29]) + res.x[pe*29+4]*(args[29] ** 0.5)
 + res.x[pe*30]*args[30] + res.x[pe*30+1]*(args[30] ** 2) + res.x[pe*30+2]*np.log(args[30]) + res.x[pe*30+3]*1/(1.0+args[30]) + res.x[pe*30+4]*(args[30] ** 0.5)
 + res.x[pe*31]*args[31] + res.x[pe*31+1]*(args[31] ** 2) + res.x[pe*31+2]*np.log(args[31]) + res.x[pe*31+3]*1/(1.0+args[31]) + res.x[pe*31+4]*(args[31] ** 0.5)
 + res.x[pe*32]*args[32] + res.x[pe*32+1]*(args[32] ** 2) + res.x[pe*32+2]*np.log(args[32]) + res.x[pe*32+3]*1/(1.0+args[32]) + res.x[pe*32+4]*(args[32] ** 0.5)
 + res.x[pe*33]*args[33] + res.x[pe*33+1]*(args[33] ** 2) + res.x[pe*33+2]*np.log(args[33]) + res.x[pe*33+3]*1/(1.0+args[33]) + res.x[pe*33+4]*(args[33] ** 0.5)
 + res.x[pe*34]*args[34] + res.x[pe*34+1]*(args[34] ** 2) + res.x[pe*34+2]*np.log(args[34]) + res.x[pe*34+3]*1/(1.0+args[34]) + res.x[pe*34+4]*(args[34] ** 0.5)
 + res.x[pe*35+0]*( args[35] - args[36] )
 + res.x[pe*35+1]*( (args[35] - args[36]) ** 2 )
 + res.x[pe*35+2]*( np.log(abs(args[35] - args[36])) )
 + res.x[pe*35+3]*( 1/(1.0+args[35]-args[36]) )
 + res.x[pe*35+4]*( abs(args[35]-args[36]) ** 0.5 )
 + res.x[pe*35+5]*( np.sin(args[35]-args[36]) )
 + res.x[pe*35+6]*( np.cos(args[35]-args[36]) )
 )

pred_ensemble = pd.DataFrame(pred_ensemble)
pred_ensemble.columns = ['loss']
print mean_absolute_error(train['loss'], pred_ensemble.values)
bound_df_retrain(pred_ensemble)
pred_ensemble.to_csv("retrain.csv")


####### prediction
args = [
    pred_nn_1['loss'],           #1
    pred_nn_2['loss'],           #2
    pred_nn_3['loss'],           #3
    pred_nn_4['loss'],           #4
    pred_nn_5['loss'],           #5
    pred_nn_6['loss'],           #6
    pred_nn_1_fix['loss'],       #7
    pred_nn_2_fix['loss'],       #8
    pred_nn_3_fix['loss'],       #9
    pred_nn_4_fix['loss'],       #10
    pred_nn_5_fix['loss'],       #11
    pred_nn_6_fix['loss'],       #12
    pred_new_nn_1['loss'],       #13
    pred_new_nn_2['loss'],       #14
    pred_new_nn_3['loss'],       #15
    pred_new_nn_4['loss'],       #16
    pred_new_nn_5['loss'],       #17
    pred_new_nn_6['loss'],       #18
    pred_new_nn_7['loss'],       #19
    pred_new_nn_1_65['loss'],    #20
    pred_new_nn_2_65['loss'],    #21
    pred_xgb_1['loss'],          #22
    pred_xgb_2['loss'],          #23
    pred_xgb_3['loss'],          #24
    pred_xgb_6['loss'],          #25
    pred_xgb_9['loss'],          #26
    pred_xgb_10['loss'],         #27
    pred_xgb_11['loss'],         #28
    pred_xgb_13['loss'],         #29
    pred_xgb_14['loss'],         #30
    pred_xgb_15['loss'],         #31
    pred_xgb_17['loss'],         #32
    pred_xgb_17_2way['loss'],    #33
    pred_xgb_18['loss'],         #34
    pred_xgb_20['loss'],         #35 ####
    pred_nn['loss'],                            #35
    pred_xgb['loss'],                           #36
]


pred_ensemble = (res.x[pe*0]*args[0] + res.x[pe*0+1]*(args[0] ** 2) + res.x[pe*0+2]*np.log(args[0]) + res.x[pe*0+3]*1/(1.0+args[0]) + res.x[pe*0+4]*(args[0] ** 0.5)
 + res.x[pe*1]*args[1] + res.x[pe*1+1]*(args[1] ** 2) + res.x[pe*1+2]*np.log(args[1]) + res.x[pe*1+3]*1/(1.0+args[1]) + res.x[pe*1+4]*(args[1] ** 0.5)
 + res.x[pe*2]*args[2] + res.x[pe*2+1]*(args[2] ** 2) + res.x[pe*2+2]*np.log(args[2]) + res.x[pe*2+3]*1/(1.0+args[2]) + res.x[pe*2+4]*(args[2] ** 0.5)
 + res.x[pe*3]*args[3] + res.x[pe*3+1]*(args[3] ** 2) + res.x[pe*3+2]*np.log(args[3]) + res.x[pe*3+3]*1/(1.0+args[3]) + res.x[pe*3+4]*(args[3] ** 0.5)
 + res.x[pe*4]*args[4] + res.x[pe*4+1]*(args[4] ** 2) + res.x[pe*4+2]*np.log(args[4]) + res.x[pe*4+3]*1/(1.0+args[4]) + res.x[pe*4+4]*(args[4] ** 0.5)
 + res.x[pe*5]*args[5] + res.x[pe*5+1]*(args[5] ** 2) + res.x[pe*5+2]*np.log(args[5]) + res.x[pe*5+3]*1/(1.0+args[5]) + res.x[pe*5+4]*(args[5] ** 0.5)
 + res.x[pe*6]*args[6] + res.x[pe*6+1]*(args[6] ** 2) + res.x[pe*6+2]*np.log(args[6]) + res.x[pe*6+3]*1/(1.0+args[6]) + res.x[pe*6+4]*(args[6] ** 0.5)
 + res.x[pe*7]*args[7] + res.x[pe*7+1]*(args[7] ** 2) + res.x[pe*7+2]*np.log(args[7]) + res.x[pe*7+3]*1/(1.0+args[7]) + res.x[pe*7+4]*(args[7] ** 0.5)
 + res.x[pe*8]*args[8] + res.x[pe*8+1]*(args[8] ** 2) + res.x[pe*8+2]*np.log(args[8]) + res.x[pe*8+3]*1/(1.0+args[8]) + res.x[pe*8+4]*(args[8] ** 0.5)
 + res.x[pe*9]*args[9] + res.x[pe*9+1]*(args[9] ** 2) + res.x[pe*9+2]*np.log(args[9]) + res.x[pe*9+3]*1/(1.0+args[9]) + res.x[pe*9+4]*(args[9] ** 0.5)
 + res.x[pe*10]*args[10] + res.x[pe*10+1]*(args[10] ** 2) + res.x[pe*10+2]*np.log(args[10]) + res.x[pe*10+3]*1/(1.0+args[10]) + res.x[pe*10+4]*(args[10] ** 0.5)
 + res.x[pe*11]*args[11] + res.x[pe*11+1]*(args[11] ** 2) + res.x[pe*11+2]*np.log(args[11]) + res.x[pe*11+3]*1/(1.0+args[11]) + res.x[pe*11+4]*(args[11] ** 0.5)
 + res.x[pe*12]*args[12] + res.x[pe*12+1]*(args[12] ** 2) + res.x[pe*12+2]*np.log(args[12]) + res.x[pe*12+3]*1/(1.0+args[12]) + res.x[pe*12+4]*(args[12] ** 0.5)
 + res.x[pe*13]*args[13] + res.x[pe*13+1]*(args[13] ** 2) + res.x[pe*13+2]*np.log(args[13]) + res.x[pe*13+3]*1/(1.0+args[13]) + res.x[pe*13+4]*(args[13] ** 0.5)
 + res.x[pe*14]*args[14] + res.x[pe*14+1]*(args[14] ** 2) + res.x[pe*14+2]*np.log(args[14]) + res.x[pe*14+3]*1/(1.0+args[14]) + res.x[pe*14+4]*(args[14] ** 0.5)
 + res.x[pe*15]*args[15] + res.x[pe*15+1]*(args[15] ** 2) + res.x[pe*15+2]*np.log(args[15]) + res.x[pe*15+3]*1/(1.0+args[15]) + res.x[pe*15+4]*(args[15] ** 0.5)
 + res.x[pe*16]*args[16] + res.x[pe*16+1]*(args[16] ** 2) + res.x[pe*16+2]*np.log(args[16]) + res.x[pe*16+3]*1/(1.0+args[16]) + res.x[pe*16+4]*(args[16] ** 0.5)
 + res.x[pe*17]*args[17] + res.x[pe*17+1]*(args[17] ** 2) + res.x[pe*17+2]*np.log(args[17]) + res.x[pe*17+3]*1/(1.0+args[17]) + res.x[pe*17+4]*(args[17] ** 0.5)
 + res.x[pe*18]*args[18] + res.x[pe*18+1]*(args[18] ** 2) + res.x[pe*18+2]*np.log(args[18]) + res.x[pe*18+3]*1/(1.0+args[18]) + res.x[pe*18+4]*(args[18] ** 0.5)
 + res.x[pe*19]*args[19] + res.x[pe*19+1]*(args[19] ** 2) + res.x[pe*19+2]*np.log(args[19]) + res.x[pe*19+3]*1/(1.0+args[19]) + res.x[pe*19+4]*(args[19] ** 0.5)
 + res.x[pe*20]*args[20] + res.x[pe*20+1]*(args[20] ** 2) + res.x[pe*20+2]*np.log(args[20]) + res.x[pe*20+3]*1/(1.0+args[20]) + res.x[pe*20+4]*(args[20] ** 0.5)
 + res.x[pe*21]*args[21] + res.x[pe*21+1]*(args[21] ** 2) + res.x[pe*21+2]*np.log(args[21]) + res.x[pe*21+3]*1/(1.0+args[21]) + res.x[pe*21+4]*(args[21] ** 0.5)
 + res.x[pe*22]*args[22] + res.x[pe*22+1]*(args[22] ** 2) + res.x[pe*22+2]*np.log(args[22]) + res.x[pe*22+3]*1/(1.0+args[22]) + res.x[pe*22+4]*(args[22] ** 0.5)
 + res.x[pe*23]*args[23] + res.x[pe*23+1]*(args[23] ** 2) + res.x[pe*23+2]*np.log(args[23]) + res.x[pe*23+3]*1/(1.0+args[23]) + res.x[pe*23+4]*(args[23] ** 0.5)
 + res.x[pe*24]*args[24] + res.x[pe*24+1]*(args[24] ** 2) + res.x[pe*24+2]*np.log(args[24]) + res.x[pe*24+3]*1/(1.0+args[24]) + res.x[pe*24+4]*(args[24] ** 0.5)
 + res.x[pe*25]*args[25] + res.x[pe*25+1]*(args[25] ** 2) + res.x[pe*25+2]*np.log(args[25]) + res.x[pe*25+3]*1/(1.0+args[25]) + res.x[pe*25+4]*(args[25] ** 0.5)
 + res.x[pe*26]*args[26] + res.x[pe*26+1]*(args[26] ** 2) + res.x[pe*26+2]*np.log(args[26]) + res.x[pe*26+3]*1/(1.0+args[26]) + res.x[pe*26+4]*(args[26] ** 0.5)
 + res.x[pe*27]*args[27] + res.x[pe*27+1]*(args[27] ** 2) + res.x[pe*27+2]*np.log(args[27]) + res.x[pe*27+3]*1/(1.0+args[27]) + res.x[pe*27+4]*(args[27] ** 0.5)
 + res.x[pe*28]*args[28] + res.x[pe*28+1]*(args[28] ** 2) + res.x[pe*28+2]*np.log(args[28]) + res.x[pe*28+3]*1/(1.0+args[28]) + res.x[pe*28+4]*(args[28] ** 0.5)
 + res.x[pe*29]*args[29] + res.x[pe*29+1]*(args[29] ** 2) + res.x[pe*29+2]*np.log(args[29]) + res.x[pe*29+3]*1/(1.0+args[29]) + res.x[pe*29+4]*(args[29] ** 0.5)
 + res.x[pe*30]*args[30] + res.x[pe*30+1]*(args[30] ** 2) + res.x[pe*30+2]*np.log(args[30]) + res.x[pe*30+3]*1/(1.0+args[30]) + res.x[pe*30+4]*(args[30] ** 0.5)
 + res.x[pe*31]*args[31] + res.x[pe*31+1]*(args[31] ** 2) + res.x[pe*31+2]*np.log(args[31]) + res.x[pe*31+3]*1/(1.0+args[31]) + res.x[pe*31+4]*(args[31] ** 0.5)
 + res.x[pe*32]*args[32] + res.x[pe*32+1]*(args[32] ** 2) + res.x[pe*32+2]*np.log(args[32]) + res.x[pe*32+3]*1/(1.0+args[32]) + res.x[pe*32+4]*(args[32] ** 0.5)
 + res.x[pe*33]*args[33] + res.x[pe*33+1]*(args[33] ** 2) + res.x[pe*33+2]*np.log(args[33]) + res.x[pe*33+3]*1/(1.0+args[33]) + res.x[pe*33+4]*(args[33] ** 0.5)
 + res.x[pe*34]*args[34] + res.x[pe*34+1]*(args[34] ** 2) + res.x[pe*34+2]*np.log(args[34]) + res.x[pe*34+3]*1/(1.0+args[34]) + res.x[pe*34+4]*(args[34] ** 0.5)
 + res.x[pe*35+0]*( args[35] - args[36] )
 + res.x[pe*35+1]*( (args[35] - args[36]) ** 2 )
 + res.x[pe*35+2]*( np.log(abs(args[35] - args[36])) )
 + res.x[pe*35+3]*( 1/(1.0+args[35]-args[36]) )
 + res.x[pe*35+4]*( abs(args[35]-args[36]) ** 0.5 )
 + res.x[pe*35+5]*( np.sin(args[35]-args[36]) )
 + res.x[pe*35+6]*( np.cos(args[35]-args[36]) )
 )


pred_ensemble = pd.DataFrame(pred_ensemble)
pred_ensemble.columns = ['loss']
bound_df_test(pred_ensemble)
pred_ensemble.to_csv("pred_retrain.csv", index_label='id')
