import os
import datetime
import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed
import time

from utilities.utils import *
from utilities.constants import *

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)

# MAPE metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

# ## Importing Scantrack data
df = pd.read_csv(os.path.join(data_dir, base_file))
del df['date']
df = df[df['channel']!='TOTAL']
df = df.rename(columns= {'date_new': 'date'})
df['date'] = pd.to_datetime(df['date'])
df.columns = ['Manufacturer', 'PPU', 'TDP', 'Units', 'Sales', 'Date', 'Channel', 'SOM']
df['Month'] = df['Date'].dt.month
# df = df[df['Date']<'2023-03-01']
max_date = df['Date'].max()
print(df.shape)

# Calculating monthly and quarterly indices
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter

df['units_avg_y'] = df.groupby(['Manufacturer','Channel','Year'])['Units'].transform('mean')
df['units_avg_q'] = df.groupby(['Manufacturer','Channel','Quarter'])['Units'].transform('mean')

df['monthly_index'] = df['Units']/df['units_avg_y']
df['quarterly_index'] = df['Units']/df['units_avg_q']

###################################   Limiting Sales Function   ###################################

df_avg = df.copy()
df_avg['Year'] = df_avg['Date'].dt.year
df_avg = df_avg.groupby(['Year', 'Manufacturer','Channel']).agg(avg_sales = ('Sales','mean'), 
                                                            avg_ppu  = ('PPU','mean'),
                                                            max_sales = ('Sales','max'),
                                                            max_ppu = ("PPU",'max')).reset_index()
df_avg['sales_dev'] = abs(df_avg['avg_sales'] - df_avg['max_sales'])/df_avg['avg_sales']
df_avg['ppu_dev'] = abs(df_avg['avg_ppu'] - df_avg['max_ppu'])/df_avg['avg_ppu']

sales_limit_df = pd.DataFrame()

for keys, df_slice in df_avg.groupby(['Channel', 'Manufacturer']):
    cnl, mfr = keys
    print(keys)
    sales_cagr, sales_dev = get_cagr(df_slice, mfr, cnl, col='avg_sales')
    ppu_cagr, ppu_dev = get_cagr(df_slice, mfr, cnl, col='avg_ppu')
    forecast_df_sales = get_est(df_slice, col='avg_sales', cagr=sales_cagr, manufacturer=mfr, channel=cnl)
    # forecast_df_sales['avg_sales'] = forecast_df_sales['avg_sales']*(1+sales_dev)
    forecast_df_ppu = get_est(df_slice, col='avg_ppu', cagr=ppu_cagr, manufacturer=mfr, channel=cnl)
    forecast_df_ppu['avg_ppu'] = forecast_df_ppu['avg_ppu']*(1+ppu_dev)

    forecast_df_sales['avg_sales'] = forecast_df_sales['avg_sales'].map(int)

    forecast_df = forecast_df_sales.merge(forecast_df_ppu, on='Year', how='left')
    forecast_df['Manufacturer'] = mfr
    forecast_df['Channel'] = cnl
    sales_limit_df = pd.concat([sales_limit_df, forecast_df])

sales_limit_df.to_csv(os.path.join(data_dir, realistic_simulation_file), index=False)

# delete previously saved models
for f in os.listdir(model_dir):
    os.remove(os.path.join(model_dir, f))

# Engineered features
avg_price_features, tdp_features, avg_price_ratio_features, tdp_ratio_features = generate_regression_features(df)
regression_features = avg_price_features + tdp_features + avg_price_ratio_features + tdp_ratio_features

# predicting TDP and PPU values for next 18 months
print('Forecasting PPU...')
df_ppu= kpi_prediction_best_model(df, 'PPU', future_months, test_months=test_months)
print('Forecasting TDP...')
df_tdp= kpi_prediction_best_model(df, 'TDP', future_months, test_months=test_months)
df_tdp['TDP_hat'] = np.ceil(df_tdp['TDP_hat'])

# predicting monthly and quarterly indices for next 18 months
df_monthly = kpi_prediction_best_model(df, 'monthly_index', future_months, test_months=test_months)
df_monthly = df_monthly.drop(columns = ['Model'])
df_quarterly = kpi_prediction_best_model(df, 'quarterly_index', future_months, test_months=test_months)
df_quarterly = df_quarterly.drop(columns = ['Model'])

df_pred_whole = df_ppu.merge(df_tdp, on = ['Manufacturer', 'Channel', 'Date'], how= 'left')
df_pred_whole = df_pred_whole.merge(df_monthly, on = ['Manufacturer', 'Channel', 'Date'], how= 'left')\
                             .merge(df_quarterly, on = ['Manufacturer', 'Channel', 'Date'], how= 'left')
df_reg_pred = df_pred_whole[df_pred_whole['Date']>max_date]
df_reg_pred = df_reg_pred.drop(columns= ['PPU', 'TDP']+seasonality_cols).rename(columns= {'PPU_hat': 'PPU', 'TDP_hat': 'TDP', 
                                                                                          'monthly_index_hat': 'monthly_index', 
                                                                                          'quarterly_index_hat': 'quarterly_index'})

# Concatenating with original data
df = pd.concat([df, df_reg_pred])
print(df.shape)
# Adding competitor price and TDP as features
df_comp = generate_comp_features_1(df)
df_comp = round(df_comp, 2)
print(df_comp.shape)

df_comp.to_csv(os.path.join(data_dir,'external_data.csv'), index=False)

start_time = time.time()
print("Forecasting Units...")
df_sales, df_feat, df_coef, df_score = sales_prediction_constrained_regression(df_comp, future_months, test_months, target= 'Units')
print("Forecasting finished in --- %s seconds ---" % (time.time() - start_time))
# to append new data to already existing model scores tracker
df_scores = pd.read_csv(os.path.join(data_dir,scores_filename))
df_scores['Date'] = pd.to_datetime(df_scores['Month'], format= "%b-%y")

# appending new data to model scores tracker
df_score['Date'] = pd.to_datetime(df_score['Month'], format= "%b-%y")

if df_scores.Date.max() < df_score.Date.max():
    df_scores = pd.concat([df_scores,df_score])
df_scores = df_scores.drop(columns = ['Date'])


# Saving files to data directory
df_sales.to_csv(os.path.join(data_dir, sales_pred_filename), index=False) 
df_feat.to_csv(os.path.join(data_dir,selected_features_filename), index=False)
df_coef.to_csv(os.path.join(data_dir, feature_importance_filename), index=False)
df_reg_pred.to_csv(os.path.join(data_dir, ppu_tdp_pred_filename), index=False)
df_scores.to_csv(os.path.join(data_dir,scores_filename), index=False)