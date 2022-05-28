import os
import numpy as np
# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

from math import ceil

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings('ignore')

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
def columns_to_str(df,columns):
    for name in columns:
        df[name] = df[name].astype(str)
    return df

def columns_to_int(df,columns):
    for name in columns:
        df[name] = df[name].astype(int)
    return df


def get_item_id(df):
    df["item_id"] = df["I100"].astype(str) +"_"+ df["C100"].astype(str) +"_"+ df["C101"].astype(str) 
    return df

def unique_comparations(training_df,submission_df,column_name,name=''):
    if name == '':
        name = column_name

    ids_submission = submission_df[column_name].unique()
    ids_training    = training_df[column_name].unique()
    print('unique '+name+' : ',len(set(ids_submission) | set(ids_training)))
    print('unknown '+name+' in submission :',set(ids_submission)-set(ids_training))
    return ids_training,ids_submission

def string_to_categorical(df):
    for name in df.columns:
        aux = df[name].dtype
        if str(aux) in ['str','object']:
            df[name] = df[name].astype('category')
            print(name,aux,df[name].dtype)
    return df

def categorical_to_numeric(df):
    cat_columns = df.select_dtypes(['category']).columns
    print(cat_columns)
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df[cat_columns] = df[cat_columns].astype(int)
    return df

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            if str(col_type) == numerics:
                c_min = df[col].min()
                c_max = df[col].max()

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            if str(col_type)[:5] == 'float':
                c_min = df[col].min()
                c_max = df[col].max()

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        

## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1



def join_columns_string(data,columns):
    auxiliar = None
    for column in columns:
        if auxiliar is None:
            auxiliar = data[column].astype(str)
        else:
            auxiliar += '_' +data[column].astype(str)        
    return auxiliar

