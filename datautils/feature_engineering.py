import os
import pandas as pd


def fe_dates(df):
    
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    
    df['year']  = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    df['day']   = df['DATE'].dt.day
    
    
    df['day_of_week'] = df['DATE'].dt.day_of_week
    df['day_of_year'] = df['DATE'].dt.day_of_year
    
    df['is_year_start']    = df['DATE'].dt.is_year_start
    df['is_quarter_start'] = df['DATE'].dt.is_quarter_start
    df['is_month_start']   = df['DATE'].dt.is_month_start
    df['is_month_end']    = df['DATE'].dt.is_month_end
    
    return df