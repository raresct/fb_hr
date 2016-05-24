#!/usr/bin/env python

import numpy as np
import pandas as pd
import xgboost as xgb

import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier

import scipy
from scipy import interp

import re
import pickle
import matplotlib.pyplot as plt

def feats_46():

    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df_sample_sub = pd.read_csv('data/sampleSubmission.csv')
    df_bids = pd.read_csv('data/bids.csv')

    feat_list = ['total_bids', 'cond', 'time_std']

    feats = [np.load('features/{}.npz'.format(fname))[fname] for fname in feat_list]

    big_x = np.hstack(feats)
    ids = df_bids['bidder_id'].unique()
    ids.sort()
    bids_train_idx = pd.DataFrame(ids).isin(list(df_train['bidder_id'])).values.reshape(-1)
    x_train = big_x[bids_train_idx,:]
    x_test = big_x[~bids_train_idx,:]
    bids_y = pd.DataFrame(ids, columns=['bidder_id']).merge(df_train, on='bidder_id')['outcome'].values
    return x_train, bids_y, x_test, ids, bids_train_idx, df_test

def feats_59():

    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df_sample_sub = pd.read_csv('data/sampleSubmission.csv')
    df_bids = pd.read_csv('data/bids.csv')

    feat_list = ['total_bids', 'cond', 'time_std', 'totals', 'bids_auction', 'time_avg']

    feats = [np.load('features/{}.npz'.format(fname))[fname] for fname in feat_list]

    big_x = np.hstack(feats)
    print(big_x.shape[1])
    ids = df_bids['bidder_id'].unique()
    ids.sort()
    bids_train_idx = pd.DataFrame(ids).isin(list(df_train['bidder_id'])).values.reshape(-1)
    x_train = big_x[bids_train_idx,:]
    x_test = big_x[~bids_train_idx,:]
    bids_y = pd.DataFrame(ids, columns=['bidder_id']).merge(df_train, on='bidder_id')['outcome'].values
    return x_train, bids_y, x_test, ids, bids_train_idx, df_test

def feats_59_poly():

    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df_sample_sub = pd.read_csv('data/sampleSubmission.csv')
    df_bids = pd.read_csv('data/bids.csv')

    feat_list = ['total_bids', 'cond', 'time_std', 'totals', 'bids_auction', 'time_avg']

    feats = [np.load('features/{}.npz'.format(fname))[fname] for fname in feat_list]

    big_x = np.hstack(feats)
    big_x = preprocessing.scale(big_x)
    poly = preprocessing.PolynomialFeatures(2)
    big_x = poly.fit_transform(big_x)
    print(big_x.shape[1])
    ids = df_bids['bidder_id'].unique()
    ids.sort()
    bids_train_idx = pd.DataFrame(ids).isin(list(df_train['bidder_id'])).values.reshape(-1)
    x_train = big_x[bids_train_idx,:]
    x_test = big_x[~bids_train_idx,:]
    bids_y = pd.DataFrame(ids, columns=['bidder_id']).merge(df_train, on='bidder_id')['outcome'].values
    return x_train, bids_y, x_test, ids, bids_train_idx, df_test

def feats_63():

    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df_sample_sub = pd.read_csv('data/sampleSubmission.csv')
    df_bids = pd.read_csv('data/bids.csv')

    feat_list = ['total_bids', 'cond', 'time_std', 'totals', 'bids_auction', 'time_avg', 'time_diff']

    feats = [np.load('features/{}.npz'.format(fname))[fname] for fname in feat_list]

    big_x = np.hstack(feats)
    print(big_x.shape[1])
    ids = df_bids['bidder_id'].unique()
    ids.sort()
    bids_train_idx = pd.DataFrame(ids).isin(list(df_train['bidder_id'])).values.reshape(-1)
    x_train = big_x[bids_train_idx,:]
    x_test = big_x[~bids_train_idx,:]
    bids_y = pd.DataFrame(ids, columns=['bidder_id']).merge(df_train, on='bidder_id')['outcome'].values
    return x_train, bids_y, x_test, ids, bids_train_idx, df_test



def feats_63_poly():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df_sample_sub = pd.read_csv('data/sampleSubmission.csv')
    df_bids = pd.read_csv('data/bids.csv')

    feat_list = ['total_bids', 'cond', 'time_std', 'totals', 'bids_auction', 'time_avg', 'time_diff']

    feats = [np.load('features/{}.npz'.format(fname))[fname] for fname in feat_list]

    big_x = np.hstack(feats)
    big_x = preprocessing.scale(big_x)
    poly = preprocessing.PolynomialFeatures(2)
    big_x = poly.fit_transform(big_x)

    print(big_x.shape[1])
    ids = df_bids['bidder_id'].unique()
    ids.sort()
    bids_train_idx = pd.DataFrame(ids).isin(list(df_train['bidder_id'])).values.reshape(-1)
    x_train = big_x[bids_train_idx,:]
    x_test = big_x[~bids_train_idx,:]
    bids_y = pd.DataFrame(ids, columns=['bidder_id']).merge(df_train, on='bidder_id')['outcome'].values
    return x_train, bids_y, x_test, ids, bids_train_idx, df_test


def make_sub(ids, bids_train_idx, df_test, bids_test_probs, sub_name):
    if len(bids_test_probs.shape) > 1:
        probs = bids_test_probs[:,1]
    else:
        probs = bids_test_probs
    submission = pd.DataFrame(np.hstack((
        ids[~bids_train_idx].reshape(-1,1),
        probs.reshape(-1,1))),
        columns=['bidder_id','prediction'])
    # add missing bidders
    missing_ids = list(set(df_test['bidder_id'])-set(submission['bidder_id']))

    missing = pd.DataFrame(np.hstack([
        np.array(missing_ids).reshape(-1,1), np.zeros((len(missing_ids),1))]),
        columns=['bidder_id','prediction'])
    submission=submission.append(missing)
    submission.to_csv(sub_name, index=False)

def svm1(x, y, x_test, seed):
    model = SVC(probability=True, random_state=seed)
    # avg CV AUC PLS
    cv = StratifiedKFold(y, n_folds=10, random_state=seed)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    #calib = CalibratedClassifierCV(model, cv=10, method='isotonic')
    for i, (train, test) in enumerate(cv):
        probas_ = model.fit(x[train], y[train]).predict_proba(x[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Training set 10CV AUC:\n{}'.format(mean_auc))
    # return probs
    model = SVC(probability=True, random_state=seed)
    model = model.fit(x, y)
    print('Training set acc:\n{}'.format(model.score(x, y)))
    bids_test_probs = model.predict_proba(x_test)
    return bids_test_probs

def svm_calib(x, y, x_test, seed):
    model = SVC(probability=True, random_state=seed)
    # avg CV AUC PLS
    cv = StratifiedKFold(y, n_folds=10, random_state=seed)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    calib = CalibratedClassifierCV(model, cv=10, method='isotonic')
    for i, (train, test) in enumerate(cv):
        probas_ = calib.fit(x[train], y[train]).predict_proba(x[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Training set 10CV AUC:\n{}'.format(mean_auc))
    # return probs
    #model = SVC(probability=True, random_state=seed)
    #model = model.fit(x, y)
    probs = np.average([cls.predict_proba(x_test) for cls in calib.calibrated_classifiers_], axis=0)
    print probs.shape
    #print('Training set acc:\n{}'.format(model2.score(x, y)))
    #bids_test_probs = model2.predict_proba(x_test)
    return probs

def svm_calib_norm(x,y,x_test,seed):
    # normalize x+x_test
    x_rows = x.shape[0]
    X = preprocessing.normalize(np.vstack((x,x_test)) , norm='l2')
    x = X[:x_rows,:]
    x_test = X[x_rows:, :]
    print x.shape
    print x_test.shape

    model = SVC(probability=True, random_state=seed)
    # avg CV AUC PLS
    cv = StratifiedKFold(y, n_folds=10, random_state=seed)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    calib = CalibratedClassifierCV(model, cv=10, method='sigmoid')
    for i, (train, test) in enumerate(cv):
        probas_ = calib.fit(x[train], y[train]).predict_proba(x[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Training set 10CV AUC:\n{}'.format(mean_auc))
    # return probs
    #model = SVC(probability=True, random_state=seed)
    #model = model.fit(x, y)
    probs = np.average([cls.predict_proba(x_test) for cls in calib.calibrated_classifiers_], axis=0)
    print probs.shape
    #print('Training set acc:\n{}'.format(model2.score(x, y)))
    #bids_test_probs = model2.predict_proba(x_test)
    return probs

def svm_calib_scale(x,y,x_test,seed):
    # normalize x+x_test
    x_rows = x.shape[0]
    X = preprocessing.scale(np.vstack((x,x_test)))
    x = X[:x_rows,:]
    x_test = X[x_rows:, :]
    print x.shape
    print x_test.shape

    model = SVC(probability=True, class_weight='auto', random_state=seed,
        C=1,gamma=0.1)
    # avg CV AUC PLS
    n_folds = 10
    cv = StratifiedKFold(y, n_folds=n_folds, random_state=seed)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    probs_list = []
    calib = CalibratedClassifierCV(model, cv=n_folds, method='isotonic')
    for i, (train, test) in enumerate(cv):
        probas_ = calib.fit(x[train], y[train]).predict_proba(x[test])
        #probas_ = model.fit(x[train], y[train]).predict_proba(x[test])
        #probs_list.append(model.predict_proba(x_test))
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Training set 10CV AUC:\n{}'.format(mean_auc))
    # return probs
    #model = SVC(probability=True, random_state=seed)
    #model = model.fit(x, y)
    #probs = np.average(probs_list, axis=0)
    probs = np.average([cls.predict_proba(x_test) for cls in calib.calibrated_classifiers_], axis=0)
    print probs.shape
    #print('Training set acc:\n{}'.format(model2.score(x, y)))
    #bids_test_probs = model2.predict_proba(x_test)
    return probs

def xgb_test(x,y,x_test,seed):
    x_rows = x.shape[0]
    X = preprocessing.scale(np.vstack((x,x_test)))
    x = X[:x_rows,:]
    x_test = X[x_rows:, :]
    
    dtrain = xgb.DMatrix( x, label=y, missing = -999.0, weight = np.ones(x.shape[0]))
    dtest = xgb.DMatrix( x_test, missing = -999.0, weight = np.ones(x_test.shape[0]))
    param = {'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'nthread':8}
    num_round = 40

    print ('running cross validation, with preprocessing function')
    # define the preprocessing function
    # used to return the preprocessed training, test data, and parameter
    # we can use this to do weight rescale, etc.
    # as a example, we try to set scale_pos_weight
    def fpreproc(dtrain, dtest, param):
        label = dtrain.get_label()
        ratio = float(np.sum(label == 0)) / np.sum(label==1)
        param['scale_pos_weight'] = ratio
        wtrain = dtrain.get_weight()
        wtest = dtest.get_weight()
        sum_weight = sum(wtrain) + sum(wtest)
        wtrain *= sum_weight / sum(wtrain)
        wtest *= sum_weight / sum(wtest)
        dtrain.set_weight(wtrain)
        dtest.set_weight(wtest)
        return (dtrain, dtest, param)

    # do cross validation, for each fold
    # the dtrain, dtest, param will be passed into fpreproc
    # then the return value of fpreproc will be used to generate
    # results of that fold
    #xgb.cv(param, dtrain, num_round, nfold=10,
    #       metrics={'ams@0.15', 'auc'}, seed = seed, fpreproc = fpreproc)
    
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    wtrain = dtrain.get_weight()
    wtest = dtest.get_weight()
    sum_weight = sum(wtrain) + sum(wtest)
    wtrain *= sum_weight / sum(wtrain)
    wtest *= sum_weight / sum(wtest)
    dtrain.set_weight(wtrain)
    dtest.set_weight(wtest)
    
    bst = xgb.train(param, dtrain, num_round)        
    
    ypred = bst.predict( dtest )
    return ypred
    
def gb_calib_scale(x,y,x_test,seed):
    # normalize x+x_test
    x_rows = x.shape[0]
    X = preprocessing.scale(np.vstack((x,x_test)))
    x = X[:x_rows,:]
    x_test = X[x_rows:, :]
    print x.shape
    print x_test.shape
    model0 = SVC(probability=True, class_weight='auto', random_state=seed,
        C=1,gamma=0.1)
    # fucking bugs in sklearn    
    class WrapClassifier:
        def __init__(self, est):
            self.est = est
        def predict(self, X):
            return self.est.predict_proba(X)[:,1][:,np.newaxis]
        def fit(self, X, y, sample_weight):
            self.est.fit(X, y, sample_weight)
    model01 = WrapClassifier(model0)            
    model1 = GradientBoostingClassifier(max_depth=3, random_state=seed, learning_rate=0.1, n_estimators=100, max_features=400)
    model = BaggingClassifier(model1, n_jobs=-1, random_state=seed)
    # avg CV AUC PLS
    n_folds = 10
    cv = StratifiedKFold(y, n_folds=n_folds, random_state=seed)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    probs_list = []
    calib = CalibratedClassifierCV(model, cv=n_folds, method='sigmoid')
    for i, (train, test) in enumerate(cv):
        probas_ = calib.fit(x[train], y[train]).predict_proba(x[test])
        #probas_ = model.fit(x[train], y[train]).predict_proba(x[test])
        #probs_list.append(model.predict_proba(x_test))
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Training set 10CV AUC:\n{}'.format(mean_auc))
    # return probs
    #model = SVC(probability=True, random_state=seed)
    #model = model.fit(x, y)
    #probs = np.average(probs_list, axis=0)
    probs = np.average([cls.predict_proba(x_test) for cls in calib.calibrated_classifiers_], axis=0)
    print probs.shape
    #print('Training set acc:\n{}'.format(model2.score(x, y)))
    #bids_test_probs = model2.predict_proba(x_test)
    return probs

def svm_rf_calib_scale(x,y,x_test,seed):
    # normalize x+x_test
    x_rows = x.shape[0]
    X = preprocessing.scale(np.vstack((x,x_test)))
    x = X[:x_rows,:]
    x_test = X[x_rows:, :]
    print x.shape
    print x_test.shape

    model = SVC(probability=True, class_weight='auto', random_state=seed,
        C=1,gamma=0.1)
    model2 = RandomForestClassifier(n_estimators=100, max_features=None,
        n_jobs=-1, class_weight='subsample', random_state=seed)
    model3 = GradientBoostingClassifier(random_state=seed)
    # avg CV AUC PLS
    n_folds = 10
    cv = StratifiedKFold(y, n_folds=n_folds, random_state=seed)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    probs_list = []
    calib = CalibratedClassifierCV(model, cv=n_folds, method='isotonic')
    calib2 = CalibratedClassifierCV(model2, cv=n_folds, method='isotonic')
    calib3 = CalibratedClassifierCV(model3, cv=n_folds, method='isotonic')
    for i, (train, test) in enumerate(cv):
        probas_1 = calib.fit(x[train], y[train]).predict_proba(x[test])
        probas_2 = calib2.fit(x[train], y[train]).predict_proba(x[test])
        probas_3 = calib3.fit(x[train], y[train]).predict_proba(x[test])
        probas_ = np.average([probas_1, probas_2, probas_3], axis=0)
        #sum_probas = np.sum(probas_[:,1])
        #probas_[:1] = probas_[:1]**3
        #probas_[:1] = probas_[:1]/np.sum(probas_[:1])*sum_probas
        
        #probas_ = model.fit(x[train], y[train]).predict_proba(x[test])
        #probs_list.append(model.predict_proba(x_test))
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Training set 10CV AUC:\n{}'.format(mean_auc))
    # return probs
    #model = SVC(probability=True, random_state=seed)
    #model = model.fit(x, y)
    #probs = np.average(probs_list, axis=0)
    probs1 = np.average([cls.predict_proba(x_test) for cls in calib.calibrated_classifiers_], axis=0)
    probs2 = np.average([cls.predict_proba(x_test) for cls in calib2.calibrated_classifiers_], axis=0)
    probs3 = np.average([cls.predict_proba(x_test) for cls in calib3.calibrated_classifiers_], axis=0)
    probs = np.average([probs1,probs2, probs3], axis=0)
    print probs.shape
    #print('Training set acc:\n{}'.format(model2.score(x, y)))
    #bids_test_probs = model2.predict_proba(x_test)
    return probs


def rf_calib_scale(x,y,x_test,seed):
    # normalize x+x_test
    x_rows = x.shape[0]
    X = preprocessing.scale(np.vstack((x,x_test)))
    x = X[:x_rows,:]
    x_test = X[x_rows:, :]
    print x.shape
    print x_test.shape

    model = RandomForestClassifier(n_estimators=100, max_features=None,
        n_jobs=-1, class_weight='subsample', random_state=seed)
    #model = SVC(probability=True, class_weight='auto', random_state=seed)
    # avg CV AUC PLS
    cv = StratifiedKFold(y, n_folds=10, random_state=seed)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    calib = CalibratedClassifierCV(model, cv=10, method='sigmoid')
    for i, (train, test) in enumerate(cv):
        probas_ = calib.fit(x[train], y[train]).predict_proba(x[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Training set 10CV AUC:\n{}'.format(mean_auc))
    # return probs
    #model = SVC(probability=True, random_state=seed)
    #model = model.fit(x, y)
    probs = np.average([cls.predict_proba(x_test) for cls in calib.calibrated_classifiers_], axis=0)
    print probs.shape
    #print('Training set acc:\n{}'.format(model2.score(x, y)))
    #bids_test_probs = model2.predict_proba(x_test)
    return probs

def svm_boost_calib_scale(x,y,x_test,seed):
    # normalize x+x_test
    x_rows = x.shape[0]
    X = preprocessing.scale(np.vstack((x,x_test)))
    x = X[:x_rows,:]
    x_test = X[x_rows:, :]
    print x.shape
    print x_test.shape

    model = SVC(probability=True, class_weight='auto', random_state=seed,
        C= 100,gamma=0.0)
    boosted = AdaBoostClassifier(model, random_state=seed)
    # avg CV AUC PLS
    cv = StratifiedKFold(y, n_folds=10, random_state=seed)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    calib = CalibratedClassifierCV(boosted, cv=10, method='isotonic')
    for i, (train, test) in enumerate(cv):
        probas_ = calib.fit(x[train], y[train]).predict_proba(x[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Training set 10CV AUC:\n{}'.format(mean_auc))
    # return probs
    #model = SVC(probability=True, random_state=seed)
    #model = model.fit(x, y)
    probs = np.average([cls.predict_proba(x_test) for cls in calib.calibrated_classifiers_], axis=0)
    print probs.shape
    #print('Training set acc:\n{}'.format(model2.score(x, y)))
    #bids_test_probs = model2.predict_proba(x_test)
    return probs

def svm_boost_isotonic_scale_train(x, y, x_test, seed):
    x_rows = x.shape[0]
    X = preprocessing.scale(np.vstack((x,x_test)))
    x = X[:x_rows,:]
    x_test = X[x_rows:, :]

    #x_tr, x_val, y_tr, y_val = train_test_split(
    #    x, y, test_size=0.1, random_state=seed)

    model = SVC(probability=True, class_weight='auto', random_state=seed)
    boosted = AdaBoostClassifier(model, random_state=seed)
    calib = CalibratedClassifierCV(boosted, cv=2, method='isotonic')
    calib.fit(x, y)
    probs = calib.predict_proba(x_test)
    return probs

def make_svm_sub():
    sub_name = 'submissions/63feats_poly_scale_gb_bag_sigmoid.csv'
    #sub_name = 'submissions/59_poly_feats_xgb.csv'
    print(sub_name)
    seed = 1234
    #np.random.seed(seed)
    x, y, x_test, ids, bids_train_idx, df_test = feats_63_poly()
    probs = gb_calib_scale(x, y, x_test, seed) #_calib_scale
    make_sub(ids, bids_train_idx, df_test, probs, sub_name)

def main():
    make_svm_sub()

if __name__=='__main__':
    main()

