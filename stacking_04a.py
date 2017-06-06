import time
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold

import model_xgb_multiclass
import model_lgbm_multiclass

OUTPUT_DIR = './layer2_04a/'

COMBS = {}

FILES = glob.glob('layer1*/*-blend_test.csv')
models_to_keep = []
for f in FILES:
    temp = f.split('-cv_')[1]
    score = float(temp.split('-')[0])
    basename = f.split('blend_test')[0]
    if (score < 0.7):
        models_to_keep.append(basename)
COMBS['stack_layer1'] = models_to_keep

def GetDataFrame(basename):
    blend_train = pd.read_csv(basename+'blend_train.csv')
    blend_train.rename(columns={'high':basename+'_high', 'medium':basename+'_medium', 'low':basename+'_low'}, inplace=True)
    
    blend_test = pd.read_csv(basename+'blend_test.csv')
    blend_test.rename(columns={'high':basename+'_high', 'medium':basename+'_medium', 'low':basename+'_low'}, inplace=True)
    
    return blend_train, blend_test

for name, files in COMBS.items():
    ti1 = time.time()
    print("\n * Stacking files '{}'".format(name))
    
    # Read the models
    train_df, test_df = GetDataFrame(files[0])
    for f in files[1:]:
        temp_tr, temp_te = GetDataFrame(f)
        train_df = pd.merge(train_df, temp_tr, on='listing_id')
        test_df = pd.merge(test_df, temp_te, on='listing_id')

    base_tr = pd.read_json('./input/train.json')
    base_te = pd.read_json('./input/test.json')
    train_df = pd.merge(train_df, base_tr[['listing_id', 'interest_level']], on='listing_id')

    magic_feat = pd.read_csv('./input/listing_image_time.csv')
    magic_feat = magic_feat.rename(columns={'Listing_Id':'listing_id', 'time_stamp':'magic_timestamp'})
    train_df = pd.merge(train_df, magic_feat, on='listing_id', how='left')
    test_df = pd.merge(test_df, magic_feat, on='listing_id', how='left')

    # Get the arrays
    target_num_map={"high":0, "medium":1, "low":2}
    train_y = np.array(train_df["interest_level"].apply(lambda x: target_num_map[x]).copy())
    train_x = train_df.drop(['interest_level'], axis=1).values
    train_id = train_df['listing_id'].values
    test_x = test_df.values
    test_id = test_df['listing_id'].values

    # Same folds that were generated in layer1
    n_folds = 5
    skf = list(StratifiedKFold(train_y, n_folds, shuffle=True, random_state=1234))

    # Second layer models definition
    xgb_p1 = {
        'objective' : 'multi:softprob',
        'silent' : 1,
        'num_class' : 3,
        'eval_metric' : "mlogloss",
        'eta' : 0.01,
        'max_depth' : 5,
        'min_child_weight' : 2,
        'subsample' : 0.8,
        'colsample_bytree' : 0.5,
        'seed' : 2017,
    }
    lgbm_p1 = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class':3,
        'metric':'multi_logloss',
        'nthread': -1,
        'silent': True,
        'num_leaves': 2**5,
        'learning_rate': 0.01,
        'max_depth': -1,
        'max_bin': 255,
        'subsample_for_bin': 50000,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.5,
        'reg_alpha': 1,
        'reg_lambda': 0,
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight': 1
    }
    clfs = {
        'lgbm_p1':model_lgbm_multiclass.Model(train_x, train_y, test_x, skf, log_loss, lgbm_p1),
        'XGB-p1':model_xgb_multiclass.Model(train_x, train_y, test_x, skf, log_loss, xgb_p1),
    }

    for modelName, clf in clfs.items():
        t21 = time.time()
        pred_y, blend_train, blend_test, score = clf.Predicting()
        out_df = pd.DataFrame(pred_y)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_id
        out_df.to_csv(OUTPUT_DIR+"{}.{}-cv_{}-{}.csv".format(name, modelName, round(score, 6), "test"), index=False)
        out_df = pd.DataFrame(blend_test)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_id
        out_df.to_csv(OUTPUT_DIR+"{}.{}-cv_{}-{}.csv".format(name, modelName, round(score, 6), "blend_test"), index=False)
        out_df = pd.DataFrame(blend_train)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = train_id
        out_df.to_csv(OUTPUT_DIR+"{}.{}-cv_{}-{}.csv".format(name, modelName, round(score, 6), "blend_train"), index=False)
        t22 = time.time()
        print("{} - score: {}\t{} min".format(modelName, score, round((t22-t21)/60.0, 2)))

    ti2 = time.time()
    print("{} min".format(round((ti2-ti1)/60.0, 2)))
