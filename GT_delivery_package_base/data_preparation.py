import os
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)

from utilities.constants import *

# looking for files in the incremental data folder
dir_list = os.listdir(incremental_data_dir)
dir_list = [x for x in dir_list if 'xls' in x]
print(f'Files found : {dir_list}')

# processing all the files in a loop
df_agg = pd.DataFrame()

for filename in dir_list:
    print(f'Processing File : {filename}')
    if 'xlsb' in filename:
        engine = 'pyxlsb'
    else:
        engine = 'openpyxl'
    sheets_dict = pd.read_excel(os.path.join(incremental_data_dir,filename), engine= engine, sheet_name=None, header= None)
    for sheetname in ['T. GT.', 'OT', 'TRADITIONAL']:
        channel = sheetname.strip().upper() 
        df = sheets_dict[sheetname].copy()
        df = df.dropna(thresh = 5).dropna(axis = 1, how='all').dropna(how='all')
        df.columns = df.iloc[0]
        df.columns = df.columns.to_series().mask(lambda x: x==np.nan).ffill()
        df = df.iloc[1:]
        df = df.T.reset_index()
        month_cols = [x for x in df.iloc[0].values if (str(x) != 'nan') and (str(x) != 'Periodo')]
        df.columns = df.iloc[0]
        df.columns = ['manufacturer', 'cols'] + month_cols
        df = df.iloc[1:]
        df = pd.melt(df, id_vars=['manufacturer', 'cols'], value_vars= month_cols, var_name= 'date')
        df = df.pivot(index=['manufacturer', 'date'], columns='cols').reset_index()
        df.columns = ['manufacturer', 'date', 'avg_price', 'tdp', 'units', 'sales']
        df['manufacturer'] = df['manufacturer'].str.upper()
        df['date_new'] = df['date'].replace(month_dict, regex=True)
        df['date_new'] = pd.to_datetime(df['date_new'], format='%B %Y')
        df['channel'] = channel
        if channel == 'T. GT.':
            df['channel'] = "TOTAL"
        else:
            df['channel'] = channel
        df_agg = pd.concat([df_agg, df])
    print(f'Finished Processing File : {filename}')

if df_agg['units'].min()>10000:
    df_agg['units'] = df_agg['units']/1000000

df_agg['sales_sum'] = df_agg.groupby(['channel', 'date_new'])['sales'].transform('sum')
df_agg['SOM'] = df_agg['sales']*100/df_agg['sales_sum']
del df_agg['sales_sum']

print(df_agg.head())

# Importing Previous Collated Data
df_sc = pd.read_csv(os.path.join(data_dir,base_file))
df_sc['date_new'] = pd.to_datetime(df_sc['date_new'])
max_date = df_sc['date_new'].max()
print(max_date)
print(df_sc.shape)

df_agg = df_agg[df_agg['date_new']>max_date]
print(df_agg.shape)
df_sc = pd.concat([df_sc, df_agg])
df_sc = df_sc.drop_duplicates()
print(df_sc.shape)

# saving the file with the same name 
df_sc.to_csv(os.path.join(data_dir, base_file), index=False)