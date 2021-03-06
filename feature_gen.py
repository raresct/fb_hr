#!/usr/bin/env python2

import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import scipy
import re
import pickle
import matplotlib.pyplot as plt

# fun of f2 per f1 for each bidder (e.g. average number of ips per auction)
def get_bidder_stat(df, fun, f1, f2):
    c1 = df.groupby(['bidder_id',f1], as_index=False)[f2].aggregate(lambda x: len(x.unique()))
    c1[f2] = c1[f2].astype('float')
    c2 = c1.groupby('bidder_id')[f2].aggregate(lambda x: fun(x) if x.size>0 else 0)
    return c2.values.reshape(-1,1)

# same as get_bidder_stat but
# accounts for missing bidder_ids
def get_bidder_stat_miss(df, fun, f1, f2):
    c1 = df.groupby(['bidder_id',f1], as_index=False)[f2].aggregate(lambda x: len(x.unique()))
    # create missing df
    ids_missing = set(df['bidder_id'])-set(c1['bidder_id'])
    len_missing = len(ids_missing)
    ids_a = np.array(list(ids_missing)).reshape(-1,1)
    empty_str_a = np.array(['' for i in range(len_missing)]).reshape(-1,1)
    missing_a = np.hstack([ids_a, empty_str_a, np.zeros((len_missing,1))])
    df_missing = pd.DataFrame(missing_a, columns=['bidder_id', f1, f2])
    # concat
    new_df = pd.concat([pd.DataFrame(c1), df_missing], ignore_index=True)
    new_df[f2] = new_df[f2].astype('float') # because pandas is fucky
    # new_df acts as c1
    c2 = new_df.groupby('bidder_id')[f2].aggregate(lambda x: fun(x) if x.size>0 else 0)
    return c2.values.reshape(-1,1)

def get_time_diff_miss(df, fun,  fun2, f1='auction', f2='time'):
    c1 = df.groupby(['bidder_id',f1], as_index=False)[f2].aggregate(lambda x: fun2(np.ediff1d(np.sort(x))) if x.size>1 else 0 )
    # create missing df
    ids_missing = set(df['bidder_id'])-set(c1['bidder_id'])
    len_missing = len(ids_missing)
    ids_a = np.array(list(ids_missing)).reshape(-1,1)
    empty_str_a = np.array(['' for i in range(len_missing)]).reshape(-1,1)
    missing_a = np.hstack([ids_a, empty_str_a, np.zeros((len_missing,1))])
    df_missing = pd.DataFrame(missing_a, columns=['bidder_id', f1, f2])
    # concat
    new_df = pd.concat([pd.DataFrame(c1), df_missing], ignore_index=True)
    new_df[f2] = new_df[f2].astype('float') # because pandas is fucky
    # new_df acts as c1
    c2 = new_df.groupby('bidder_id')[f2].aggregate(lambda x: fun(x) if x.size>0 else 0)
    return c2.values.reshape(-1,1)

# convert categorical features to vectors
def df_cat(df, col_names):
    le = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder()
    col_l = [cat_proc(df, name, le, ohe) for name in col_names]
    return scipy.sparse.hstack(col_l)

def cat_proc(df, name, le, ohe):
    col_arr = df[name].values
    col_int = le.fit_transform(col_arr)
    col_int = col_int.reshape(col_int.shape[0],1)
    col_cat = ohe.fit_transform(col_int)
    print col_cat.shape
    return col_cat

class FeatureGen:
    def __init__(self):
        self.df_train = pd.read_csv('data/train.csv')
        self.df_test = pd.read_csv('data/test.csv')
        self.df_sample_sub = pd.read_csv('data/sampleSubmission.csv')
        self.df_bids = pd.read_csv('data/bids.csv')
        
    def total_bids(self):
        # Feature 1: total bids per bidder_id
        total_bids = (self.df_bids.groupby('bidder_id').count()['bid_id']
            .values.reshape(-1,1))
        print total_bids.shape
        print np.sum(np.isnan(total_bids))
        np.savez('total_bids.npz', **{'total_bids':total_bids})
        
    # Features (4-32): for each f1 and f2 in country, ip, device, auction, url
    # compute mean and std
    def conditional_mean_std_for_pairs(self):
        f_d = {}
        # this should be done using Cartesian products
        # s1 = country, ip, device, auction, url
        # for el1,el2,el3 in cartesian_product([s1,s1,['np.mean','np.std']]):
        #   f_d[(el1,el2,el3)] = get_bidder_stat_miss(self.df_bids, el3, el1, el2)
        
        # auction feats
        f_d[('auction','country','mean')] = get_bidder_stat(self.df_bids, np.mean, 
            'auction', 'country')
        f_d[('auction','country','std')] = get_bidder_stat(self.df_bids, np.std, 
            'auction', 'country')
        f_d[('auction','ip','mean')] = get_bidder_stat(self.df_bids, np.mean, 
            'auction', 'ip')
        f_d[('auction','ip','std')] = get_bidder_stat(self.df_bids, np.std, 
            'auction', 'ip')
        f_d[('auction','url','mean')] = get_bidder_stat(self.df_bids, np.mean, 
            'auction', 'url')
        f_d[('auction','url','std')] = get_bidder_stat(self.df_bids, np.std, 
            'auction', 'url')
        f_d[('auction','device','mean')] = get_bidder_stat(self.df_bids, np.mean, 
            'auction', 'device')
        f_d[('auction','device','std')] = get_bidder_stat(self.df_bids, np.std, 
            'auction', 'device')        
         # country feats        
        f_d[('country','auction','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'country', 'auction')
        f_d[('country','auction','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'country', 'auction')
        f_d[('country','ip','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'country', 'ip')
        f_d[('country','ip','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'country', 'ip')
        f_d[('country','url','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'country', 'url')
        f_d[('country','url','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'country', 'url')
        f_d[('country','device','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'country', 'device')
        f_d[('country','device','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'country', 'device')
        # url feats
        f_d[('url','auction','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'url', 'auction')
        f_d[('url','auction','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'url', 'auction')
        f_d[('url','ip','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'url', 'ip')
        f_d[('url','ip','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'url', 'ip')
        f_d[('url','country','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'url', 'country')
        f_d[('url','country','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'url', 'country')
        f_d[('url','device','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'url', 'device')
        f_d[('url','device','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'url', 'device')
        # ip feats
        f_d[('ip','auction','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'ip', 'auction')
        f_d[('ip','auction','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'ip', 'auction')
        f_d[('ip','url','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'ip', 'url')
        f_d[('ip','url','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'ip', 'url')
        f_d[('ip','country','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'ip', 'country')
        f_d[('ip','country','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'ip', 'country')
        f_d[('ip','device','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'ip', 'device')
        f_d[('ip','device','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'ip', 'device')    
         # device feats
        f_d[('device','auction','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'device', 'auction')
        f_d[('device','auction','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'device', 'auction')
        f_d[('device','url','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'device', 'url')
        f_d[('device','url','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'device', 'url')
        f_d[('device','country','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'device', 'country')
        f_d[('device','country','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'device', 'country')
        f_d[('device','ip','mean')] = get_bidder_stat_miss(self.df_bids, np.mean, 
            'device', 'ip')
        f_d[('device','ip','std')] = get_bidder_stat_miss(self.df_bids, np.std, 
            'device', 'ip')        
                    
        cond = np.hstack([v for _,v in f_d.iteritems() ])
        print cond.shape
        print np.sum(np.isnan(total_bids))
        np.savez('cond.npz', **{'cond':cond})
     
    def conditional_time_std(self):
        f_d = {}
        el2 = 'time'
        for el in ['auction', 'device', 'country', 'ip', 'url']:
            f_d[(el, el2, np.std)] = get_bidder_stat_miss(self.df_bids, np.std, 
                el, el2)
        cond = np.hstack([v for _,v in f_d.iteritems() ])
        print cond.shape
        print np.sum(np.isnan(cond))
        np.savez('time_std.npz', **{'time_std':cond})

    def direct(self):
        all_names = ['auction', 'merchandise', 'device', 'country', 'url']
        bids_cat = df_cat(df_bids, all_names)
        bids_sparse = scipy.sparse.hstack([bids_cat, df_bids['time'].values.reshape(-1,1)])
        df_bids_feats = bids_sparse.toarray()
        
    def totals(self):
        totals = np.hstack([self.df_bids.groupby('bidder_id').count()[el]
            .values.reshape(-1,1) 
            for el in ['auction', 'merchandise', 'device', 'country', 'url', 'ip']])
        print totals.shape
        print np.sum(np.isnan(totals))
        np.savez('totals.npz', **{'totals':totals})
    
    def bids_auction(self):
        x1 = get_bidder_stat_miss(self.df_bids, np.mean, 
            'auction', 'bid_id')
        x2 = get_bidder_stat_miss(self.df_bids, np.std, 
            'auction', 'bid_id')    
        bids_auction = np.hstack([x1,x2])
        print bids_auction.shape
        print np.sum(np.isnan(bids_auction))
        np.savez('bids_auction.npz', **{'bids_auction':bids_auction})
        
    def conditional_time_avg(self):
        f_d = {}
        el2 = 'time'
        for el in ['auction', 'device', 'country', 'ip', 'url']:
            f_d[(el, el2, np.std)] = get_bidder_stat_miss(self.df_bids, np.mean, 
                el, el2)
        cond = np.hstack([v for _,v in f_d.iteritems() ])
        print cond.shape
        print np.sum(np.isnan(cond))
        np.savez('time_avg.npz', **{'time_avg':cond})
    
    def time_diff(self):
        time = self.df_bids['time'].values
        time = time.astype(np.float64)
        std_time = np.std(time, dtype=np.float64)
        time -= np.mean(time, dtype=np.float64)
        time /= std_time
        time = preprocessing.scale(time)
        self.df_bids['time'] = time
        x1 = get_time_diff_miss(self.df_bids, np.mean, np.mean)
        x2 = get_time_diff_miss(self.df_bids, np.mean, np.std)
        x3 = get_time_diff_miss(self.df_bids, np.std, np.mean)
        x4 = get_time_diff_miss(self.df_bids, np.std, np.std)
        time_diff = np.hstack([x1,x2,x3,x4])
        #time_diff = np.nan_to_num(time_diff)
        print time_diff.shape
        print np.sum(np.isnan(time_diff))
        np.savez('time_diff.npz', **{'time_diff':time_diff})
        
def main():
    fg = FeatureGen()
    #fg.total_bids()
    #fg.conditional_mean_std_for_pairs()
    #fg.conditional_time_std()
    #fg.totals()
    #fg.bids_auction()
    #fg.conditional_time_avg()
    fg.time_diff()
    
if __name__=='__main__':
    main()    
    
    
    
    
    
    
