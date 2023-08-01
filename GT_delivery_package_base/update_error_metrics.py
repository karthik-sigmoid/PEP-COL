import os
import datetime
import pandas as pd
import numpy as np
import joblib
from dateutil.relativedelta import relativedelta

from utilities.constants import *
from utilities.utils import *

import warnings
warnings.filterwarnings('ignore')
# This code updates the error metrics files with the new data 
# this is to be done every time there is a new data file in the incremental data folder

# reading scantrack file
df = pd.read_csv(os.path.join(data_dir, base_file))

# reading sales_prediction file
df_sales_pred = pd.read_csv(os.path.join(data_dir, sales_pred_filename))

# reading sales/SOM and TDP/PPU error metrics files
df_sales_som_error = pd.read_csv(os.path.join(data_dir, sales_req_file))
df_ppu_tdp_error = pd.read_csv(os.path.join(data_dir, ppu_tdp_req_file))

# Converting date into a datetime object
df['date_new'] = pd.to_datetime(df['date_new'])
df_sales_pred['Date'] = pd.to_datetime(df_sales_pred['Date'])
df_sales_som_error['Date'] = pd.to_datetime(df_sales_som_error['Date'])
df_sales_som_error = df_sales_som_error.rename(columns= {'Sales_pred': 'sales_pred', 'Sales MAPE': 'sales MAPE'})
df_ppu_tdp_error['Date'] = pd.to_datetime(df_ppu_tdp_error['Date'])

# Calculating dates of existing data and predictions start data
data_till = df_sales_pred[~df_sales_pred['Sales'].isna()]['Date'].max()
predictions_start_date = df_sales_pred[df_sales_pred['Sales'].isna()]['Date'].min()
append_data_till = predictions_start_date + relativedelta(months=11)
print(df['date_new'].max(), predictions_start_date)
# only predictions data
if df['date_new'].max() == predictions_start_date:
    print('Updating Error Matrices ...')
    df_pred_only = df_sales_pred[df_sales_pred['Sales'].isna()]
    df_pred_only = df_pred_only[(df_pred_only['Date']<=append_data_till)]
    remove_data_from = predictions_start_date - relativedelta(months=12)
    df_pred_only['forecast_month'] = pd.to_datetime(data_till).strftime("%B %Y")

    df_sales_som_final = pd.concat([df_sales_som_error,df_pred_only[['Manufacturer',
                                        'Channel',
                                        'Date',
                                        'sales_pred','units_pred',
                                        'SOM_hat','forecast_month']]])

    df_actual = df.copy()
    df_actual.rename(columns = {"date_new":"Date", 
                                'avg_price':'PPU',
                                'tdp':'TDP',
                                'sales':'Sales',
                                'units':"Units", # check for units column
                                'manufacturer':'Manufacturer', 
                                'channel':'Channel'}, inplace=True)

    df_sales_som_final = pd.merge(df_sales_som_final.drop(columns=['Sales','SOM','Units']), 
                                df_actual[['Manufacturer',
                                            'Channel',
                                            'Date',
                                            'Sales',
                                            'SOM',
                                            "Units"]], 
                                on = ['Manufacturer','Channel','Date'], how='left')

    df_sales_som_final = df_sales_som_final[df_sales_som_final['forecast_month']!=pd.to_datetime(remove_data_from).strftime("%B %Y")]

    df_tdp_ppu_final = pd.concat([df_ppu_tdp_error,df_pred_only[['Manufacturer','Channel',
                                        'Date','PPU','TDP',
                                                'forecast_month']].rename(columns = {"PPU":"PPU_hat",
                                                                                "TDP":"TDP_hat"})])

    df_tdp_ppu_final = pd.merge(df_tdp_ppu_final.drop(columns=['TDP','PPU']), df_actual[['Manufacturer',
                                                                                        'Channel','Date',
                                                                                        'PPU','TDP']], 
                        on = ['Manufacturer','Channel','Date'], how='left')

    df_tdp_ppu_final = df_tdp_ppu_final[df_tdp_ppu_final['forecast_month']!=pd.to_datetime(remove_data_from).strftime("%B %Y")]

    df_sales_som_final['sales MAPE'] = np.abs(df_sales_som_final['Sales'] -
                                            df_sales_som_final['sales_pred'])*100/np.abs(df_sales_som_final['Sales'])
    df_sales_som_final['units MAPE'] = np.abs(df_sales_som_final['Units'] -
                                            df_sales_som_final['units_pred'])*100/np.abs(df_sales_som_final['Sales'])
    
    df_sales_som_final['SOM mae'] = np.abs(df_sales_som_final['SOM']-df_sales_som_final['SOM_hat'])


    df_tdp_ppu_final['TDP MAPE'] = np.abs(df_tdp_ppu_final['TDP'] - df_tdp_ppu_final['TDP_hat'])*100/np.abs(df_tdp_ppu_final['TDP'])
    df_tdp_ppu_final['PPU MAPE'] = np.abs(df_tdp_ppu_final['PPU'] - df_tdp_ppu_final['PPU_hat'])*100/np.abs(df_tdp_ppu_final['PPU'])

    df_res_som, df_res_sales, df_res_units = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_res_ppu, df_res_tdp = pd.DataFrame(), pd.DataFrame()

    for mnth in df_sales_som_final['forecast_month'].unique():
        df_sales_som_final2 = df_sales_som_final[df_sales_som_final['forecast_month'] == mnth].dropna(subset=['Sales']).reset_index(drop=True)
        df_tdp_ppu_final2 = df_tdp_ppu_final[df_tdp_ppu_final['forecast_month'] == mnth].dropna(subset=['PPU']).reset_index(drop=True)
        df_res_som, df_res_sales, df_res_units = som_sales_table(df_sales_som_final2, df_res_som, df_res_sales, df_res_units)
        df_res_ppu, df_res_tdp = tdp_ppu_table(df_tdp_ppu_final2, df_res_ppu, df_res_tdp)

    # Rounding the values to 2 decimal places
    df_res_ppu = round(df_res_ppu,2)
    df_res_tdp = round(df_res_tdp,2)
    df_tdp_ppu_final = round(df_tdp_ppu_final, 2)
    df_sales_som_final = round(df_sales_som_final, 2)
    df_res_som = round(df_res_som, 2)
    df_res_sales = round(df_res_sales, 2)
    df_res_units = round(df_res_units,2)

    # Renaming columns
    df_res_ppu.rename(columns = {'training_till' : 'Training Upto', 'overall_wape' : 'Overall WAPE'}, inplace = True)
    df_res_tdp.rename(columns = {'training_till' : 'Training Upto', 'overall_wape' : 'Overall WAPE'}, inplace = True)
    df_sales_som_final.rename(columns = {'sales_pred' : 'Sales_pred', 'sales MAPE' : 'Sales MAPE'}, inplace = True)
    df_res_sales.rename(columns = {'training_till' : 'Training Upto', 'overall_wape' : 'Overall WAPE'}, inplace = True)
    df_res_units.rename(columns = {'training_till' : 'Training Upto', 'overall_wape' : 'Overall WAPE'}, inplace = True)
    df_res_som.rename(columns = {'training_till' : 'Training Upto', 'overall_mae' : 'Overall MAE'}, inplace = True)

    df_sales_som_final.to_csv(os.path.join(data_dir, sales_req_file), index=False)
    df_tdp_ppu_final.to_csv(os.path.join(data_dir, ppu_tdp_req_file), index = False)
    df_res_sales.to_csv(os.path.join(data_dir, sales_error_file), index=False)
    df_res_units.to_csv(os.path.join(data_dir, units_error_file), index=False)
    df_res_som.to_csv(os.path.join(data_dir, som_error_file), index=False)
    df_res_ppu.to_csv(os.path.join(data_dir, ppu_error_file), index=False)
    df_res_tdp.to_csv(os.path.join(data_dir, tdp_error_file), index=False)