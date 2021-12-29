import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
from pandas.plotting import scatter_matrix

import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn import preprocessing

def mark_missing(df):
    nan_is_0 = ["TEMP","HEART_RATE","GLUCOSE","SAT_O2","BLOOD_PRES_SYS","BLOOD_PRES_DIAS"]
    df[nan_is_0] = df[nan_is_0].replace(0,np.nan)
    return df

def drop_not_understandable(df):
    df.pop("GLUCOSE")
    df.pop("DESTINATION")
    return df

def impute_missing(df):
    cat_mask = (df.dtypes == 'object')
    cat_cols = df.columns[cat_mask].tolist()

    df_cat = df[cat_cols]
    df_num = df.drop(cat_cols, axis = 1)
    
    #CATEGORICAL
    df_cat.head()
    imp_cat = SimpleImputer(strategy = 'most_frequent')
    df_cat = pd.DataFrame(imp_cat.fit_transform(df_cat), index = df_cat.index, columns = df_cat.columns)
    ohe = preprocessing.OneHotEncoder(sparse = False)
    df_cat_ohe = pd.DataFrame(ohe.fit_transform(df_cat),
                          index = df_cat.index, columns = ohe.get_feature_names(df_cat.columns.tolist()))
    df_cat_ohe.head()
    
    #NUMERICAL
    imp_num = SimpleImputer(strategy = 'mean') # although variables are continuous only integer values. Can use mean?
    df_num = pd.DataFrame(imp_num.fit_transform(df_num), index = df_num.index, columns = df_num.columns)
    
    df_preprocessed = pd.merge(left = df_cat_ohe, right = df_num, on = 'ID')
    return df_preprocessed

def remove_outliers(df):
    for index, row in df.iterrows():
        if row.AGE > 125 or row.HEART_RATE > 200 or row.SAT_O2 < 20 or row.BLOOD_PRES_SYS > 300 or row.BLOOD_PRES_DIAS > 300:
            df.drop(index,axis=0,inplace=True)
    
    covmat = df.cov()
    covmat_inv = np.linalg.inv(covmat)
    
    avg_vect = df.mean().to_numpy()
    idx_list = []
    dist_list = []
    for index, row in df.iterrows():
        idx_list.append(index)    
        dist_list.append(((row.to_numpy()-avg_vect).dot(covmat_inv)).dot(row.to_numpy()-avg_vect))
    
    limit = np.quantile(dist_list,0.95)
    
    for i in range(len(idx_list)):
        if dist_list[i] > limit:
            df.drop(idx_list[i],axis=0,inplace=True) 
    
    return df

def scale_data(df):
    columns = df.columns
    index = df.index
    mms = preprocessing.MinMaxScaler() #MinMaxScaler to keep dummy vars
    df = mms.fit_transform(df)
    df = pd.DataFrame(df, index = index, columns = columns)
    return df

def discretize_data(df, k):
    columns = df.columns
    index = df.index
    binner = preprocessing.KBinsDiscretizer(n_bins = k, encode = "onehot-dense")
    df = pd.DataFrame(binner.fit_transform(df))
    return df

def preprocess_data(df, scaler, k):
    df = mark_missing(df)
    df = drop_not_understandable(df)
    df = impute_missing(df)
    # df = remove_outliers(df)
    if scaler == 1: # only scale if scaler = 1
        df = scale_data(df)
    if k > 1: # if 0 or 1 no discretizing. Otherwise number is the number of bins
        df = discretize_data(df,k)
    return df




