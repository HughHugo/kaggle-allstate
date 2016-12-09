# -*- coding: utf-8 -*-
import sys
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
pred_new_nn_5_10bag_retrain = pd.read_csv('../../NEW_NN_5_10bag/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_10bag_retrain)
pred_new_nn_5_double_retrain = pd.read_csv('../../NEW_NN_5_double/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_double_retrain)
pred_new_nn_5_interaction_65_retrain = pd.read_csv('../../NEW_NN_5_interaction_65/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_interaction_65_retrain)
pred_new_nn_5_interaction_70_retrain = pd.read_csv('../../NEW_NN_5_interaction_70/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_interaction_70_retrain)
pred_new_nn_5_interaction_80_retrain = pd.read_csv('../../NEW_NN_5_interaction_80/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_interaction_80_retrain)
pred_new_nn_5_interaction_90_retrain = pd.read_csv('../../NEW_NN_5_interaction_90/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_interaction_90_retrain)
pred_new_nn_5_20_shift_retrain = pd.read_csv('../../NEW_NN_5_20_SHIFT/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_20_shift_retrain)
pred_new_nn_5_100_shift_retrain = pd.read_csv('../../NEW_NN_5_100_SHIFT/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_100_shift_retrain)
pred_new_nn_5_2000_shift_retrain = pd.read_csv('../../NEW_NN_5_2000_SHIFT/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_2000_shift_retrain)
pred_new_nn_5_20000_shift_retrain = pd.read_csv('../../NEW_NN_5_20000_SHIFT/NN_retrain_5.csv', index_col=0)
bound_df_retrain(pred_new_nn_5_20000_shift_retrain)


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
#
pred_xgb_14_retrain = pd.read_csv('../../XGB_14/XGB_retrain_14.csv', index_col=0)
bound_df_retrain(pred_xgb_14_retrain)
pred_xgb_14_obj_retrain = pd.read_csv('../../XGB_14_obj/XGB_retrain_2.csv', index_col=0)
bound_df_retrain(pred_xgb_14_obj_retrain)
pred_xgb_14_obj_2_fast_retrain = pd.read_csv('../../XGB_14_obj_2_fast/XGB_retrain_2.csv', index_col=0)
bound_df_retrain(pred_xgb_14_obj_2_fast_retrain)
#
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
pred_xgb_21_retrain = pd.read_csv('../../XGB_21/XGB_retrain_21.csv', index_col=0)
bound_df_retrain(pred_xgb_21_retrain)
pred_xgb_22_retrain = pd.read_csv('../../XGB_22/XGB_retrain_22.csv', index_col=0)
bound_df_retrain(pred_xgb_22_retrain)
#
pred_xgb_24_retrain = pd.read_csv('../../XGB_24/XGB_retrain_24.csv', index_col=0)
bound_df_retrain(pred_xgb_24_retrain)
pred_xgb_24_obj_2_retrain = pd.read_csv('../../XGB_24_obj_2/XGB_retrain_2.csv', index_col=0)
bound_df_retrain(pred_xgb_24_obj_2_retrain)
pred_xgb_24_obj_fast_retrain = pd.read_csv('../../XGB_24_obj_fast/XGB_retrain_2.csv', index_col=0)
bound_df_retrain(pred_xgb_24_obj_fast_retrain)
#
pred_xgb_25_retrain = pd.read_csv('../../XGB_25/XGB_retrain_25.csv', index_col=0)
bound_df_retrain(pred_xgb_25_retrain)
pred_xgb_25_obj_fast_retrain = pd.read_csv('../../XGB_25_obj_fast/XGB_retrain_2.csv', index_col=0)
bound_df_retrain(pred_xgb_25_obj_fast_retrain)
pred_xgb_25_obj_2_fast_retrain = pd.read_csv('../../XGB_25_obj_2_fast/XGB_retrain_2.csv', index_col=0)
bound_df_retrain(pred_xgb_25_obj_2_fast_retrain)
#
pred_r_xgb_1_retrain = pd.read_csv('../../R_XGB_1/XGB_retrain_1.csv', index_col=0)
bound_df_retrain(pred_r_xgb_1_retrain)


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


pred_new_nn_5_10bag = pd.read_csv('../../NEW_NN_5_10bag/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5_10bag)
pred_new_nn_5_double = pd.read_csv('../../NEW_NN_5_double/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5_double)
pred_new_nn_5_interaction_65 = pd.read_csv('../../NEW_NN_5_interaction_65/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5_interaction_65)
pred_new_nn_5_interaction_70 = pd.read_csv('../../NEW_NN_5_interaction_70/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5_interaction_70)
pred_new_nn_5_interaction_80 = pd.read_csv('../../NEW_NN_5_interaction_80/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5_interaction_80)
pred_new_nn_5_interaction_90 = pd.read_csv('../../NEW_NN_5_interaction_90/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5_interaction_90)
pred_new_nn_5_20_shift = pd.read_csv('../../NEW_NN_5_20_SHIFT/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5_20_shift)
pred_new_nn_5_100_shift = pd.read_csv('../../NEW_NN_5_100_SHIFT/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5_100_shift)
pred_new_nn_5_2000_shift = pd.read_csv('../../NEW_NN_5_2000_SHIFT/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5_2000_shift)
pred_new_nn_5_20000_shift = pd.read_csv('../../NEW_NN_5_20000_SHIFT/NN_5.csv', index_col=0)
bound_df_test(pred_new_nn_5_20000_shift)


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
#
pred_xgb_14 = pd.read_csv('../../XGB_14/XGB_14.csv', index_col=0)
bound_df_test(pred_xgb_14)
pred_xgb_14_obj = pd.read_csv('../../XGB_14_obj/XGB_2.csv', index_col=0)
bound_df_test(pred_xgb_14_obj)
pred_xgb_14_obj_2_fast = pd.read_csv('../../XGB_14_obj_2_fast/XGB_2.csv', index_col=0)
bound_df_test(pred_xgb_14_obj_2_fast)
#
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
pred_xgb_21 = pd.read_csv('../../XGB_21/XGB_21.csv', index_col=0)
bound_df_test(pred_xgb_21)
pred_xgb_22 = pd.read_csv('../../XGB_22/XGB_22.csv', index_col=0)
bound_df_test(pred_xgb_22)
#
pred_xgb_24 = pd.read_csv('../../XGB_24/XGB_24.csv', index_col=0)
bound_df_test(pred_xgb_24)
pred_xgb_24_obj_2 = pd.read_csv('../../XGB_24_obj_2/XGB_2.csv', index_col=0)
bound_df_test(pred_xgb_24_obj_2)
pred_xgb_24_obj_fast = pd.read_csv('../../XGB_24_obj_fast/XGB_2.csv', index_col=0)
bound_df_test(pred_xgb_24_obj_fast)
#
pred_xgb_25 = pd.read_csv('../../XGB_25/XGB_25.csv', index_col=0)
bound_df_test(pred_xgb_25)
pred_xgb_25_obj_fast = pd.read_csv('../../XGB_25_obj_fast/XGB_2.csv', index_col=0)
bound_df_test(pred_xgb_25_obj_fast)
pred_xgb_25_obj_2_fast = pd.read_csv('../../XGB_25_obj_2_fast/XGB_2.csv', index_col=0)
bound_df_test(pred_xgb_25_obj_2_fast)
#
pred_r_xgb_1 = pd.read_csv('../../R_XGB_1/XGB_1.csv', index_col=0)
bound_df_test(pred_r_xgb_1)


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
print mean_absolute_error(train['loss'], pred_new_nn_5_10bag_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_5_double_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_5_interaction_65_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_5_interaction_70_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_5_interaction_80_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_5_interaction_90_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_5_2000_shift_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_5_20000_shift_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_5_2000_shift_retrain)
print mean_absolute_error(train['loss'], pred_new_nn_5_20000_shift_retrain)
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
                pred_new_nn_5_10bag_retrain.values,
                pred_new_nn_5_double_retrain.values,
                pred_new_nn_5_interaction_65_retrain.values,
                pred_new_nn_5_interaction_70_retrain.values,
                pred_new_nn_5_interaction_80_retrain.values,
                pred_new_nn_5_interaction_90_retrain.values,
                pred_new_nn_5_20_shift_retrain.values,
                pred_new_nn_5_100_shift_retrain.values,
                pred_new_nn_5_2000_shift_retrain.values,
                pred_new_nn_5_20000_shift_retrain.values,
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
             pred_new_nn_5_10bag['loss'],
             pred_new_nn_5_double['loss'],
             pred_new_nn_5_interaction_65['loss'],
             pred_new_nn_5_interaction_70['loss'],
             pred_new_nn_5_interaction_80['loss'],
             pred_new_nn_5_interaction_90['loss'],
             pred_new_nn_5_20_shift['loss'],
             pred_new_nn_5_100_shift['loss'],
             pred_new_nn_5_2000_shift['loss'],
             pred_new_nn_5_20000_shift['loss'],
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
print mean_absolute_error(train['loss'], pred_xgb_14_obj_retrain)
print mean_absolute_error(train['loss'], pred_xgb_14_obj_2_fast_retrain)
print mean_absolute_error(train['loss'], pred_xgb_15_retrain)
print mean_absolute_error(train['loss'], pred_xgb_17_retrain)
print mean_absolute_error(train['loss'], pred_xgb_17_2way_retrain)
print mean_absolute_error(train['loss'], pred_xgb_18_retrain)
print mean_absolute_error(train['loss'], pred_xgb_19_retrain)
print mean_absolute_error(train['loss'], pred_xgb_20_retrain)
print mean_absolute_error(train['loss'], pred_xgb_21_retrain)
print mean_absolute_error(train['loss'], pred_xgb_22_retrain)
print mean_absolute_error(train['loss'], pred_xgb_24_retrain)
print mean_absolute_error(train['loss'], pred_xgb_24_obj_2_retrain)
print mean_absolute_error(train['loss'], pred_xgb_24_obj_fast_retrain)
print mean_absolute_error(train['loss'], pred_xgb_25_retrain)
print mean_absolute_error(train['loss'], pred_xgb_25_obj_fast_retrain)
print mean_absolute_error(train['loss'], pred_xgb_25_obj_2_fast_retrain)
print mean_absolute_error(train['loss'], pred_r_xgb_1_retrain)


xgb_retrain_pool = [pred_xgb_1_retrain.values,
                pred_xgb_2_retrain.values,
                pred_xgb_3_retrain.values,
                pred_xgb_6_retrain.values,
                pred_xgb_9_retrain.values,
                pred_xgb_10_retrain.values,
                pred_xgb_11_retrain.values,
                pred_xgb_13_retrain.values,
                pred_xgb_14_retrain.values,
                pred_xgb_14_obj_retrain.values,
                pred_xgb_14_obj_2_fast_retrain.values,
                pred_xgb_15_retrain.values,
                pred_xgb_17_retrain.values,
                pred_xgb_17_2way_retrain.values,
                pred_xgb_18_retrain.values,
                pred_xgb_19_retrain.values,
                pred_xgb_20_retrain.values,
                pred_xgb_21_retrain.values,
                pred_xgb_22_retrain.values,
                pred_xgb_24_retrain.values,
                pred_xgb_24_obj_2_retrain.values,
                pred_xgb_24_obj_fast_retrain.values,
                pred_xgb_25_retrain.values,
                pred_xgb_25_obj_fast_retrain.values,
                pred_xgb_25_obj_2_fast_retrain.values,
                pred_r_xgb_1_retrain.values,
                ]

print "#"
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
              pred_xgb_14_obj['loss'],
              pred_xgb_14_obj_2_fast['loss'],
              pred_xgb_15['loss'],
              pred_xgb_17['loss'],
              pred_xgb_17_2way['loss'],
              pred_xgb_18['loss'],
              pred_xgb_19['loss'],
              pred_xgb_20['loss'],
              pred_xgb_21['loss'],
              pred_xgb_22['loss'],
              pred_xgb_24['loss'],
              pred_xgb_24_obj_2['loss'],
              pred_xgb_24_obj_fast['loss'],
              pred_xgb_25['loss'],
              pred_xgb_25_obj_fast['loss'],
              pred_xgb_25_obj_2_fast['loss'],
              pred_r_xgb_1['loss'],
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
    pred_nn_1_retrain['loss'].values,                     #1
    pred_nn_2_retrain['loss'].values,                     #2
    pred_nn_3_retrain['loss'].values,                     #3
    pred_nn_4_retrain['loss'].values,                     #4
    pred_nn_5_retrain['loss'].values,                     #5
    pred_nn_6_retrain['loss'].values,                     #6
    pred_nn_1_fix_retrain['loss'].values,                 #7
    pred_nn_2_fix_retrain['loss'].values,                 #8
    pred_nn_3_fix_retrain['loss'].values,                 #9
    pred_nn_4_fix_retrain['loss'].values,                 #10
    pred_nn_5_fix_retrain['loss'].values,                 #11
    pred_nn_6_fix_retrain['loss'].values,                 #12
    pred_new_nn_1_retrain['loss'].values,                 #13
    pred_new_nn_2_retrain['loss'].values,                 #14
    pred_new_nn_3_retrain['loss'].values,                 #15
    pred_new_nn_4_retrain['loss'].values,                 #16
    pred_new_nn_5_retrain['loss'].values,                 #17
    pred_new_nn_5_10bag_retrain['loss'].values,           #18
    pred_new_nn_5_double_retrain['loss'].values,          #19
    pred_new_nn_5_interaction_65_retrain['loss'].values,  #20
    pred_new_nn_5_interaction_70_retrain['loss'].values,  #21
    pred_new_nn_5_interaction_80_retrain['loss'].values,  #22
    pred_new_nn_5_interaction_90_retrain['loss'].values,  #23
    pred_new_nn_5_20_shift_retrain['loss'].values,        #24
    pred_new_nn_5_100_shift_retrain['loss'].values,       #25
    pred_new_nn_5_2000_shift_retrain['loss'].values,      #26
    pred_new_nn_5_20000_shift_retrain['loss'].values,     #27
    pred_new_nn_6_retrain['loss'].values,                 #28
    pred_new_nn_7_retrain['loss'].values,                 #29
    pred_new_nn_1_65_retrain['loss'].values,              #30
    pred_new_nn_2_65_retrain['loss'].values,              #31
    pred_xgb_1_retrain['loss'].values,                    #32
    pred_xgb_2_retrain['loss'].values,                    #33
    pred_xgb_3_retrain['loss'].values,                    #34
    pred_xgb_6_retrain['loss'].values,                    #35
    pred_xgb_9_retrain['loss'].values,                    #36
    pred_xgb_10_retrain['loss'].values,                   #37
    pred_xgb_11_retrain['loss'].values,                   #38
    pred_xgb_13_retrain['loss'].values,                   #39
    pred_xgb_14_retrain['loss'].values,                   #40
    pred_xgb_14_obj_retrain['loss'].values,               #41
    pred_xgb_14_obj_2_fast_retrain['loss'].values,        #42
    pred_xgb_15_retrain['loss'].values,                   #43
    pred_xgb_17_retrain['loss'].values,                   #44
    pred_xgb_17_2way_retrain['loss'].values,              #45
    pred_xgb_18_retrain['loss'].values,                   #46
    pred_xgb_19_retrain['loss'].values,                   #47
    pred_xgb_20_retrain['loss'].values,                   #48
    pred_xgb_21_retrain['loss'].values,                   #49
    pred_xgb_22_retrain['loss'].values,                   #50
    pred_xgb_24_retrain['loss'].values,                   #51
    pred_xgb_24_obj_2_retrain['loss'].values,             #52
    pred_xgb_24_obj_fast_retrain['loss'].values,          #53
    pred_xgb_25_retrain['loss'].values,                   #54
    pred_xgb_25_obj_fast_retrain['loss'].values,          #55
    pred_xgb_25_obj_2_fast_retrain['loss'].values,        #56
    pred_r_xgb_1_retrain['loss'].values,                  #57 ####
    pred_nn_retrain['loss'].values,                       #58 14
    pred_xgb_retrain['loss'].values,                      #59 13
    pred_nn_retrain_gmean['loss'].values,                 #60 12
    pred_xgb_retrain_gmean['loss'].values,                #61 11
    pred_nn_retrain_sd['loss'].values,                    #62 10
    pred_xgb_retrain_sd['loss'].values,                   #63 9
    pred_nn_retrain_mean_sd_1['loss'].values,             #64 8
    pred_xgb_retrain_mean_sd_1['loss'].values,            #65 7
    pred_nn_retrain_mean_sd_2['loss'].values,             #66 6
    pred_xgb_retrain_mean_sd_2['loss'].values,            #67 5
    pred_nn_retrain_mean_sd_3['loss'].values,             #68 4
    pred_xgb_retrain_mean_sd_3['loss'].values,            #69 3
    pred_nn_retrain_mean_sd_4['loss'].values,             #70 2
    pred_xgb_retrain_mean_sd_4['loss'].values,            #71 1
    train['loss'].values
]

print len(args)-1-14
pe= 5

def f(coord,args):
    return np.mean( np.abs(
       coord[pe*0]*args[0] + coord[pe*0+1]*(args[0] ** 2) + coord[pe*0+2]*np.log(args[0]) + coord[pe*0+3]*1/(1.0+args[0]) + coord[pe*0+4]*(args[0] ** 0.5)
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
     + coord[pe*36]*args[36] + coord[pe*36+1]*(args[36] ** 2) + coord[pe*36+2]*np.log(args[36]) + coord[pe*36+3]*1/(1.0+args[36]) + coord[pe*36+4]*(args[36] ** 0.5)
     + coord[pe*37]*args[37] + coord[pe*37+1]*(args[37] ** 2) + coord[pe*37+2]*np.log(args[37]) + coord[pe*37+3]*1/(1.0+args[37]) + coord[pe*37+4]*(args[37] ** 0.5)
     + coord[pe*38]*args[38] + coord[pe*38+1]*(args[38] ** 2) + coord[pe*38+2]*np.log(args[38]) + coord[pe*38+3]*1/(1.0+args[38]) + coord[pe*38+4]*(args[38] ** 0.5)
     + coord[pe*39]*args[39] + coord[pe*39+1]*(args[39] ** 2) + coord[pe*39+2]*np.log(args[39]) + coord[pe*39+3]*1/(1.0+args[39]) + coord[pe*39+4]*(args[39] ** 0.5)
     + coord[pe*40]*args[40] + coord[pe*40+1]*(args[40] ** 2) + coord[pe*40+2]*np.log(args[40]) + coord[pe*40+3]*1/(1.0+args[40]) + coord[pe*40+4]*(args[40] ** 0.5)
     + coord[pe*41]*args[41] + coord[pe*41+1]*(args[41] ** 2) + coord[pe*41+2]*np.log(args[41]) + coord[pe*41+3]*1/(1.0+args[41]) + coord[pe*41+4]*(args[41] ** 0.5)
     + coord[pe*42]*args[42] + coord[pe*42+1]*(args[42] ** 2) + coord[pe*42+2]*np.log(args[42]) + coord[pe*42+3]*1/(1.0+args[42]) + coord[pe*42+4]*(args[42] ** 0.5)
     + coord[pe*43]*args[43] + coord[pe*43+1]*(args[43] ** 2) + coord[pe*43+2]*np.log(args[43]) + coord[pe*43+3]*1/(1.0+args[43]) + coord[pe*43+4]*(args[43] ** 0.5)
     + coord[pe*44]*args[44] + coord[pe*44+1]*(args[44] ** 2) + coord[pe*44+2]*np.log(args[44]) + coord[pe*44+3]*1/(1.0+args[44]) + coord[pe*44+4]*(args[44] ** 0.5)
     + coord[pe*45]*args[45] + coord[pe*45+1]*(args[45] ** 2) + coord[pe*45+2]*np.log(args[45]) + coord[pe*45+3]*1/(1.0+args[45]) + coord[pe*45+4]*(args[45] ** 0.5)
     + coord[pe*46]*args[46] + coord[pe*46+1]*(args[46] ** 2) + coord[pe*46+2]*np.log(args[46]) + coord[pe*46+3]*1/(1.0+args[46]) + coord[pe*46+4]*(args[46] ** 0.5)
     + coord[pe*47]*args[47] + coord[pe*47+1]*(args[47] ** 2) + coord[pe*47+2]*np.log(args[37]) + coord[pe*47+3]*1/(1.0+args[47]) + coord[pe*47+4]*(args[47] ** 0.5)
     + coord[pe*48]*args[48] + coord[pe*48+1]*(args[48] ** 2) + coord[pe*48+2]*np.log(args[48]) + coord[pe*48+3]*1/(1.0+args[48]) + coord[pe*48+4]*(args[48] ** 0.5)
     + coord[pe*49]*args[49] + coord[pe*49+1]*(args[49] ** 2) + coord[pe*49+2]*np.log(args[49]) + coord[pe*49+3]*1/(1.0+args[49]) + coord[pe*49+4]*(args[49] ** 0.5)
     + coord[pe*50]*args[50] + coord[pe*50+1]*(args[50] ** 2) + coord[pe*50+2]*np.log(args[50]) + coord[pe*50+3]*1/(1.0+args[50]) + coord[pe*50+4]*(args[50] ** 0.5)
     + coord[pe*51]*args[51] + coord[pe*51+1]*(args[51] ** 2) + coord[pe*51+2]*np.log(args[51]) + coord[pe*51+3]*1/(1.0+args[51]) + coord[pe*51+4]*(args[51] ** 0.5)
     + coord[pe*52]*args[52] + coord[pe*52+1]*(args[52] ** 2) + coord[pe*52+2]*np.log(args[52]) + coord[pe*52+3]*1/(1.0+args[52]) + coord[pe*52+4]*(args[52] ** 0.5)
     + coord[pe*53]*args[53] + coord[pe*53+1]*(args[53] ** 2) + coord[pe*53+2]*np.log(args[53]) + coord[pe*53+3]*1/(1.0+args[53]) + coord[pe*53+4]*(args[53] ** 0.5)
     + coord[pe*54]*args[54] + coord[pe*54+1]*(args[54] ** 2) + coord[pe*54+2]*np.log(args[54]) + coord[pe*54+3]*1/(1.0+args[54]) + coord[pe*54+4]*(args[54] ** 0.5)
     + coord[pe*55]*args[55] + coord[pe*55+1]*(args[55] ** 2) + coord[pe*55+2]*np.log(args[55]) + coord[pe*55+3]*1/(1.0+args[55]) + coord[pe*55+4]*(args[55] ** 0.5)
     + coord[pe*56]*args[56] + coord[pe*56+1]*(args[56] ** 2) + coord[pe*56+2]*np.log(args[56]) + coord[pe*56+3]*1/(1.0+args[56]) + coord[pe*56+4]*(args[56] ** 0.5)
     #+ coord[pe*57]*args[57] + coord[pe*57+1]*(args[57] ** 2) + coord[pe*57+2]*np.log(args[57]) + coord[pe*57+3]*1/(1.0+args[57]) + coord[pe*57+4]*(args[57] ** 0.5)
     #+ coord[pe*58]*args[58] + coord[pe*58+1]*(args[58] ** 2) + coord[pe*58+2]*np.log(args[58]) + coord[pe*58+3]*1/(1.0+args[58]) + coord[pe*58+4]*(args[58] ** 0.5)
     + coord[pe*57+0]*( args[-15] - args[-14] )
     + coord[pe*57+1]*( (args[-15] - args[-14]) ** 2 )
     + coord[pe*57+2]*( np.log(abs(args[-15] - args[-14])) )
     + coord[pe*57+3]*( 1/(1.0+args[-15]-args[-14]) )
     + coord[pe*57+4]*( abs(args[-15]-args[-14]) ** 0.5 )
     + coord[pe*57+5]*( np.sin(args[-15]-args[-14]) )
     + coord[pe*57+6]*( np.cos(args[-15]-args[-14]) )
     + coord[pe*57+7]*( args[-13] - args[-12] )
     + coord[pe*57+8]*( args[-11] - args[-10] )
     + coord[pe*57+9]*( args[-9] - args[-8] )
     + coord[pe*57+10]*( args[-7] - args[-6] )
     + coord[pe*57+11]*( args[-5] - args[-4] )
     + coord[pe*57+12]*( args[-3] - args[-2] )
     - args[-1]
     ))

#SEED = 6174
constant_seed = 4
try:
    SEED = 6174 + int(sys.argv[1])
    np.random.seed(SEED) #3
    print "#1"
except:
    print "#2"
    SEED = 6174 + constant_seed
    np.random.seed(SEED)

initial_guess = np.array([np.random.uniform(-0.01,0.01) for x in range(pe * 58 + 13)])


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

pred_ensemble = (
   res.x[pe*0]*args[0] + res.x[pe*0+1]*(args[0] ** 2) + res.x[pe*0+2]*np.log(args[0]) + res.x[pe*0+3]*1/(1.0+args[0]) + res.x[pe*0+4]*(args[0] ** 0.5)
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
 + res.x[pe*35]*args[35] + res.x[pe*35+1]*(args[35] ** 2) + res.x[pe*35+2]*np.log(args[35]) + res.x[pe*35+3]*1/(1.0+args[35]) + res.x[pe*35+4]*(args[35] ** 0.5)
 + res.x[pe*36]*args[36] + res.x[pe*36+1]*(args[36] ** 2) + res.x[pe*36+2]*np.log(args[36]) + res.x[pe*36+3]*1/(1.0+args[36]) + res.x[pe*36+4]*(args[36] ** 0.5)
 + res.x[pe*37]*args[37] + res.x[pe*37+1]*(args[37] ** 2) + res.x[pe*37+2]*np.log(args[37]) + res.x[pe*37+3]*1/(1.0+args[37]) + res.x[pe*37+4]*(args[37] ** 0.5)
 + res.x[pe*38]*args[38] + res.x[pe*38+1]*(args[38] ** 2) + res.x[pe*38+2]*np.log(args[38]) + res.x[pe*38+3]*1/(1.0+args[38]) + res.x[pe*38+4]*(args[38] ** 0.5)
 + res.x[pe*39]*args[39] + res.x[pe*39+1]*(args[39] ** 2) + res.x[pe*39+2]*np.log(args[39]) + res.x[pe*39+3]*1/(1.0+args[39]) + res.x[pe*39+4]*(args[39] ** 0.5)
 + res.x[pe*40]*args[40] + res.x[pe*40+1]*(args[40] ** 2) + res.x[pe*40+2]*np.log(args[40]) + res.x[pe*40+3]*1/(1.0+args[40]) + res.x[pe*40+4]*(args[40] ** 0.5)
 + res.x[pe*41]*args[41] + res.x[pe*41+1]*(args[41] ** 2) + res.x[pe*41+2]*np.log(args[41]) + res.x[pe*41+3]*1/(1.0+args[41]) + res.x[pe*41+4]*(args[41] ** 0.5)
 + res.x[pe*42]*args[42] + res.x[pe*42+1]*(args[42] ** 2) + res.x[pe*42+2]*np.log(args[42]) + res.x[pe*42+3]*1/(1.0+args[42]) + res.x[pe*42+4]*(args[42] ** 0.5)
 + res.x[pe*43]*args[43] + res.x[pe*43+1]*(args[43] ** 2) + res.x[pe*43+2]*np.log(args[43]) + res.x[pe*43+3]*1/(1.0+args[43]) + res.x[pe*43+4]*(args[43] ** 0.5)
 + res.x[pe*44]*args[44] + res.x[pe*44+1]*(args[44] ** 2) + res.x[pe*44+2]*np.log(args[44]) + res.x[pe*44+3]*1/(1.0+args[44]) + res.x[pe*44+4]*(args[44] ** 0.5)
 + res.x[pe*45]*args[45] + res.x[pe*45+1]*(args[45] ** 2) + res.x[pe*45+2]*np.log(args[45]) + res.x[pe*45+3]*1/(1.0+args[45]) + res.x[pe*45+4]*(args[45] ** 0.5)
 + res.x[pe*46]*args[46] + res.x[pe*46+1]*(args[46] ** 2) + res.x[pe*46+2]*np.log(args[46]) + res.x[pe*46+3]*1/(1.0+args[46]) + res.x[pe*46+4]*(args[46] ** 0.5)
 + res.x[pe*47]*args[47] + res.x[pe*47+1]*(args[47] ** 2) + res.x[pe*47+2]*np.log(args[37]) + res.x[pe*47+3]*1/(1.0+args[47]) + res.x[pe*47+4]*(args[47] ** 0.5)
 + res.x[pe*48]*args[48] + res.x[pe*48+1]*(args[48] ** 2) + res.x[pe*48+2]*np.log(args[48]) + res.x[pe*48+3]*1/(1.0+args[48]) + res.x[pe*48+4]*(args[48] ** 0.5)
 + res.x[pe*49]*args[49] + res.x[pe*49+1]*(args[49] ** 2) + res.x[pe*49+2]*np.log(args[49]) + res.x[pe*49+3]*1/(1.0+args[49]) + res.x[pe*49+4]*(args[49] ** 0.5)
 + res.x[pe*50]*args[50] + res.x[pe*50+1]*(args[50] ** 2) + res.x[pe*50+2]*np.log(args[50]) + res.x[pe*50+3]*1/(1.0+args[50]) + res.x[pe*50+4]*(args[50] ** 0.5)
 + res.x[pe*51]*args[51] + res.x[pe*51+1]*(args[51] ** 2) + res.x[pe*51+2]*np.log(args[51]) + res.x[pe*51+3]*1/(1.0+args[51]) + res.x[pe*51+4]*(args[51] ** 0.5)
 + res.x[pe*52]*args[52] + res.x[pe*52+1]*(args[52] ** 2) + res.x[pe*52+2]*np.log(args[52]) + res.x[pe*52+3]*1/(1.0+args[52]) + res.x[pe*52+4]*(args[52] ** 0.5)
 + res.x[pe*53]*args[53] + res.x[pe*53+1]*(args[53] ** 2) + res.x[pe*53+2]*np.log(args[53]) + res.x[pe*53+3]*1/(1.0+args[53]) + res.x[pe*53+4]*(args[53] ** 0.5)
 + res.x[pe*54]*args[54] + res.x[pe*54+1]*(args[54] ** 2) + res.x[pe*54+2]*np.log(args[54]) + res.x[pe*54+3]*1/(1.0+args[54]) + res.x[pe*54+4]*(args[54] ** 0.5)
 + res.x[pe*55]*args[55] + res.x[pe*55+1]*(args[55] ** 2) + res.x[pe*55+2]*np.log(args[55]) + res.x[pe*55+3]*1/(1.0+args[55]) + res.x[pe*55+4]*(args[55] ** 0.5)
 + res.x[pe*56]*args[56] + res.x[pe*56+1]*(args[56] ** 2) + res.x[pe*56+2]*np.log(args[56]) + res.x[pe*56+3]*1/(1.0+args[56]) + res.x[pe*56+4]*(args[56] ** 0.5)
 + res.x[pe*57]*args[57] + res.x[pe*57+1]*(args[57] ** 2) + res.x[pe*57+2]*np.log(args[57]) + res.x[pe*57+3]*1/(1.0+args[57]) + res.x[pe*57+4]*(args[57] ** 0.5)
 #+ res.x[pe*58]*args[58] + res.x[pe*58+1]*(args[58] ** 2) + res.x[pe*58+2]*np.log(args[58]) + res.x[pe*58+3]*1/(1.0+args[58]) + res.x[pe*58+4]*(args[58] ** 0.5)
 + res.x[pe*47+0]*( args[-15] - args[-14] )
 + res.x[pe*47+1]*( (args[-15] - args[-14]) ** 2 )
 + res.x[pe*47+2]*( np.log(abs(args[-15] - args[-14])) )
 + res.x[pe*47+3]*( 1/(1.0+args[-15]-args[-14]) )
 + res.x[pe*47+4]*( abs(args[-15]-args[-14]) ** 0.5 )
 + res.x[pe*47+5]*( np.sin(args[-15]-args[-14]) )
 + res.x[pe*47+6]*( np.cos(args[-15]-args[-14]) )
 + res.x[pe*47+7]*( args[-13] - args[-12] )
 + res.x[pe*47+8]*( args[-11] - args[-10] )
 + res.x[pe*47+9]*( args[-9] - args[-8] )
 + res.x[pe*47+10]*( args[-7] - args[-6] )
 + res.x[pe*47+11]*( args[-5] - args[-4] )
 + res.x[pe*47+12]*( args[-3] - args[-2] )
 )

pred_ensemble = pd.DataFrame(pred_ensemble)
pred_ensemble.columns = ['loss']
bound_df_retrain(pred_ensemble)
print mean_absolute_error(train['loss'], pred_ensemble.values)

pred_ensemble.to_csv("retrain_%s.csv"%SEED)


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
    pred_new_nn_5_10bag['loss'], #18
    pred_new_nn_5_double['loss'],        #19
    pred_new_nn_5_interaction_65['loss'],#20
    pred_new_nn_5_interaction_70['loss'],#21
    pred_new_nn_5_20_shift['loss'],    #22
    pred_new_nn_5_100_shift['loss'],   #23
    pred_new_nn_5_2000_shift['loss'],    #22
    pred_new_nn_5_20000_shift['loss'],   #23
    pred_new_nn_6['loss'],       #24
    pred_new_nn_7['loss'],       #25
    pred_new_nn_1_65['loss'],    #26
    pred_new_nn_2_65['loss'],    #27
    pred_xgb_1['loss'],          #28
    pred_xgb_2['loss'],          #29
    pred_xgb_3['loss'],          #30
    pred_xgb_6['loss'],          #31
    pred_xgb_9['loss'],          #32
    pred_xgb_10['loss'],         #33
    pred_xgb_11['loss'],         #34
    pred_xgb_13['loss'],         #35
    #pred_xgb_14['loss'],         #36
    pred_xgb_14_obj['loss'],
    pred_xgb_14_obj_2['loss'],
    pred_xgb_14_obj_fast['loss'],
    pred_xgb_14_obj_2_fast['loss'],
    pred_xgb_15['loss'],         #37
    pred_xgb_17['loss'],         #38
    pred_xgb_17_2way['loss'],    #39
    pred_xgb_18['loss'],         #40
    pred_xgb_19['loss'],         #41
    pred_xgb_20['loss'],         #42
    pred_xgb_21['loss'],         #43
    pred_xgb_22['loss'],         #44
    #pred_xgb_24['loss'],         #45
    pred_xgb_24_obj['loss'],
    pred_xgb_24_obj_2['loss'],
    pred_xgb_24_obj_fast['loss'],
    pred_xgb_24_obj_2_fast['loss'],
    #pred_xgb_25['loss'],         #46
    pred_xgb_25_obj['loss'],
    pred_xgb_25_obj_2['loss'],
    pred_xgb_25_obj_fast['loss'],
    pred_xgb_25_obj_2_fast['loss'],
    pred_r_xgb_1['loss'],        #47 ####
    pred_nn['loss'],             #48 14
    pred_xgb['loss'],            #49 13
    pred_nn_gmean['loss'],       #50 12
    pred_xgb_gmean['loss'],      #51 11
    pred_nn_sd['loss'],          #52 10
    pred_xgb_sd['loss'],         #53 9
    pred_nn_mean_sd_1['loss'],   #54 8
    pred_xgb_mean_sd_1['loss'],  #55 7
    pred_nn_mean_sd_2['loss'],   #56 6
    pred_xgb_mean_sd_2['loss'],  #57 5
    pred_nn_mean_sd_3['loss'],   #58 4
    pred_xgb_mean_sd_3['loss'],  #59 3
    pred_nn_mean_sd_4['loss'],   #60 2
    pred_xgb_mean_sd_4['loss'],  #61 1
    train['loss']
]



pred_ensemble = (
       res.x[pe*0]*args[0] + res.x[pe*0+1]*(args[0] ** 2) + res.x[pe*0+2]*np.log(args[0]) + res.x[pe*0+3]*1/(1.0+args[0]) + res.x[pe*0+4]*(args[0] ** 0.5)
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
     + res.x[pe*35]*args[35] + res.x[pe*35+1]*(args[35] ** 2) + res.x[pe*35+2]*np.log(args[35]) + res.x[pe*35+3]*1/(1.0+args[35]) + res.x[pe*35+4]*(args[35] ** 0.5)
     + res.x[pe*36]*args[36] + res.x[pe*36+1]*(args[36] ** 2) + res.x[pe*36+2]*np.log(args[36]) + res.x[pe*36+3]*1/(1.0+args[36]) + res.x[pe*36+4]*(args[36] ** 0.5)
     + res.x[pe*37]*args[37] + res.x[pe*37+1]*(args[37] ** 2) + res.x[pe*37+2]*np.log(args[37]) + res.x[pe*37+3]*1/(1.0+args[37]) + res.x[pe*37+4]*(args[37] ** 0.5)
     + res.x[pe*38]*args[38] + res.x[pe*38+1]*(args[38] ** 2) + res.x[pe*38+2]*np.log(args[38]) + res.x[pe*38+3]*1/(1.0+args[38]) + res.x[pe*38+4]*(args[38] ** 0.5)
     + res.x[pe*39]*args[39] + res.x[pe*39+1]*(args[39] ** 2) + res.x[pe*39+2]*np.log(args[39]) + res.x[pe*39+3]*1/(1.0+args[39]) + res.x[pe*39+4]*(args[39] ** 0.5)
     + res.x[pe*40]*args[40] + res.x[pe*40+1]*(args[40] ** 2) + res.x[pe*40+2]*np.log(args[40]) + res.x[pe*40+3]*1/(1.0+args[40]) + res.x[pe*40+4]*(args[40] ** 0.5)
     + res.x[pe*41]*args[41] + res.x[pe*41+1]*(args[41] ** 2) + res.x[pe*41+2]*np.log(args[41]) + res.x[pe*41+3]*1/(1.0+args[41]) + res.x[pe*41+4]*(args[41] ** 0.5)
     + res.x[pe*42]*args[42] + res.x[pe*42+1]*(args[42] ** 2) + res.x[pe*42+2]*np.log(args[42]) + res.x[pe*42+3]*1/(1.0+args[42]) + res.x[pe*42+4]*(args[42] ** 0.5)
     + res.x[pe*43]*args[43] + res.x[pe*43+1]*(args[43] ** 2) + res.x[pe*43+2]*np.log(args[43]) + res.x[pe*43+3]*1/(1.0+args[43]) + res.x[pe*43+4]*(args[43] ** 0.5)
     + res.x[pe*44]*args[44] + res.x[pe*44+1]*(args[44] ** 2) + res.x[pe*44+2]*np.log(args[44]) + res.x[pe*44+3]*1/(1.0+args[44]) + res.x[pe*44+4]*(args[44] ** 0.5)
     + res.x[pe*45]*args[45] + res.x[pe*45+1]*(args[45] ** 2) + res.x[pe*45+2]*np.log(args[45]) + res.x[pe*45+3]*1/(1.0+args[45]) + res.x[pe*45+4]*(args[45] ** 0.5)
     + res.x[pe*46]*args[46] + res.x[pe*46+1]*(args[46] ** 2) + res.x[pe*46+2]*np.log(args[46]) + res.x[pe*46+3]*1/(1.0+args[46]) + res.x[pe*46+4]*(args[46] ** 0.5)
     + res.x[pe*47]*args[47] + res.x[pe*47+1]*(args[47] ** 2) + res.x[pe*47+2]*np.log(args[37]) + res.x[pe*47+3]*1/(1.0+args[47]) + res.x[pe*47+4]*(args[47] ** 0.5)
     + res.x[pe*48]*args[48] + res.x[pe*48+1]*(args[48] ** 2) + res.x[pe*48+2]*np.log(args[48]) + res.x[pe*48+3]*1/(1.0+args[48]) + res.x[pe*48+4]*(args[48] ** 0.5)
     + res.x[pe*49]*args[49] + res.x[pe*49+1]*(args[49] ** 2) + res.x[pe*49+2]*np.log(args[49]) + res.x[pe*49+3]*1/(1.0+args[49]) + res.x[pe*49+4]*(args[49] ** 0.5)
     + res.x[pe*50]*args[50] + res.x[pe*50+1]*(args[50] ** 2) + res.x[pe*50+2]*np.log(args[50]) + res.x[pe*50+3]*1/(1.0+args[50]) + res.x[pe*50+4]*(args[50] ** 0.5)
     + res.x[pe*51]*args[51] + res.x[pe*51+1]*(args[51] ** 2) + res.x[pe*51+2]*np.log(args[51]) + res.x[pe*51+3]*1/(1.0+args[51]) + res.x[pe*51+4]*(args[51] ** 0.5)
     + res.x[pe*52]*args[52] + res.x[pe*52+1]*(args[52] ** 2) + res.x[pe*52+2]*np.log(args[52]) + res.x[pe*52+3]*1/(1.0+args[52]) + res.x[pe*52+4]*(args[52] ** 0.5)
     + res.x[pe*53]*args[53] + res.x[pe*53+1]*(args[53] ** 2) + res.x[pe*53+2]*np.log(args[53]) + res.x[pe*53+3]*1/(1.0+args[53]) + res.x[pe*53+4]*(args[53] ** 0.5)
     + res.x[pe*54]*args[54] + res.x[pe*54+1]*(args[54] ** 2) + res.x[pe*54+2]*np.log(args[54]) + res.x[pe*54+3]*1/(1.0+args[54]) + res.x[pe*54+4]*(args[54] ** 0.5)
     + res.x[pe*55]*args[55] + res.x[pe*55+1]*(args[55] ** 2) + res.x[pe*55+2]*np.log(args[55]) + res.x[pe*55+3]*1/(1.0+args[55]) + res.x[pe*55+4]*(args[55] ** 0.5)
     + res.x[pe*56]*args[56] + res.x[pe*56+1]*(args[56] ** 2) + res.x[pe*56+2]*np.log(args[56]) + res.x[pe*56+3]*1/(1.0+args[56]) + res.x[pe*56+4]*(args[56] ** 0.5)
     + res.x[pe*57]*args[57] + res.x[pe*57+1]*(args[57] ** 2) + res.x[pe*57+2]*np.log(args[57]) + res.x[pe*57+3]*1/(1.0+args[57]) + res.x[pe*57+4]*(args[57] ** 0.5)
     #+ res.x[pe*58]*args[58] + res.x[pe*58+1]*(args[58] ** 2) + res.x[pe*58+2]*np.log(args[58]) + res.x[pe*58+3]*1/(1.0+args[58]) + res.x[pe*58+4]*(args[58] ** 0.5)
     + res.x[pe*47+0]*( args[-15] - args[-14] )
     + res.x[pe*47+1]*( (args[-15] - args[-14]) ** 2 )
     + res.x[pe*47+2]*( np.log(abs(args[-15] - args[-14])) )
     + res.x[pe*47+3]*( 1/(1.0+args[-15]-args[-14]) )
     + res.x[pe*47+4]*( abs(args[-15]-args[-14]) ** 0.5 )
     + res.x[pe*47+5]*( np.sin(args[-15]-args[-14]) )
     + res.x[pe*47+6]*( np.cos(args[-15]-args[-14]) )
     + res.x[pe*47+7]*( args[-13] - args[-12] )
     + res.x[pe*47+8]*( args[-11] - args[-10] )
     + res.x[pe*47+9]*( args[-9] - args[-8] )
     + res.x[pe*47+10]*( args[-7] - args[-6] )
     + res.x[pe*47+11]*( args[-5] - args[-4] )
     + res.x[pe*47+12]*( args[-3] - args[-2] )
 )


pred_ensemble = pd.DataFrame(pred_ensemble)
pred_ensemble.columns = ['loss']
bound_df_test(pred_ensemble)
pred_ensemble.to_csv("pred_retrain_%s.csv"%SEED, index_label='id')