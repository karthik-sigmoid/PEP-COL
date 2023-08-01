import datetime

from utilities.constants import *
from utilities.utils import *

import warnings
warnings.filterwarnings('ignore')


# Creating empty dataframes for storing model performances
df_res_som, df_res_sales, df_res_units = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
df_res_ppu, df_res_tdp = pd.DataFrame(), pd.DataFrame()

# Importing Scantrack Base Data
df = pd.read_csv(os.path.join(data_dir, base_file))
del df['date']
df = df[df['channel']!='TOTAL']
df = df.rename(columns= {'date_new': 'date'})
df['date'] = pd.to_datetime(df['date'])
df.columns = ['Manufacturer', 'PPU', 'TDP', 'Units', 'Sales', 'Date', 'Channel', 'SOM']
df['Month'] = df['Date'].dt.month
max_date = df['Date'].max()
print(df.shape)

# Calculating monthly and quarterly indices
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter

df['units_avg_y'] = df.groupby(['Manufacturer','Channel','Year'])['Units'].transform('mean')
df['units_avg_q'] = df.groupby(['Manufacturer','Channel','Quarter'])['Units'].transform('mean')

df['monthly_index'] = df['Units']/df['units_avg_y']
df['quarterly_index'] = df['Units']/df['units_avg_q']

# Extracting generated feature names
avg_price_features, tdp_features, avg_price_ratio_features, tdp_ratio_features = generate_regression_features(df)
regression_features = avg_price_features + tdp_features + avg_price_ratio_features + tdp_ratio_features


## Iterating over number of months for which training and testing to be done and to generate model performance matrices
# Empty dataframes to store next 12 months predictions after each iteration
df_req = pd.DataFrame()
df_req_tdp_ppu = pd.DataFrame()
df_scores = pd.DataFrame()

for i in np.arange(11, 0, -1):
    n_days, future_months = i, 12
    print(n_days)
    
    # Generate PPU and TDP predictions for future months
    df_avg_price = kpi_prediction_best_model(df, 'PPU', future_months,n_days=n_days)
    df_tdp= kpi_prediction_best_model(df, 'TDP', future_months,n_days=n_days)
    df_tdp['TDP_hat'] = np.ceil(df_tdp['TDP_hat'])
    df_monthly = kpi_prediction_best_model(df, 'monthly_index', future_months, n_days=n_days)
    df_monthly = df_monthly.drop(columns = ['Model'])
    df_quarterly = kpi_prediction_best_model(df, 'quarterly_index', future_months, n_days=n_days)
    df_quarterly = df_quarterly.drop(columns = ['Model'])
    df_pred_whole = df_avg_price.merge(df_tdp, on = ['Manufacturer', 'Channel', 'Date'], how= 'left')
    df_pred_whole = df_pred_whole.merge(df_monthly, on = ['Manufacturer', 'Channel', 'Date'], how= 'left')\
                             .merge(df_quarterly, on = ['Manufacturer', 'Channel', 'Date'], how= 'left')

    # Storing only future months predictions in a separate dataframe
    df_track3 = pd.DataFrame()
    cols = ['Manufacturer', 'Channel', 'Date', 'TDP', 'TDP_hat', 'PPU', 'PPU_hat']
    for keys, df_slice in df_pred_whole.groupby(['Channel', 'Manufacturer']):
        df_slice = df_slice.sort_values(by='Date')[cols].tail(future_months)
        df_track3 = pd.concat([df_track3, df_slice])

    # Storing dataframes till actual data is available and generating performance matrix for PPU and TDP
    df_track4 = pd.DataFrame()
    for keys, df_slice in df_track3.groupby(['Manufacturer', 'Channel']):
        df_slice['TDP MAPE'] = np.abs(df_slice['TDP'] - df_slice['TDP_hat'])*100/np.abs(df_slice['TDP'])
        df_slice['PPU MAPE'] = np.abs(df_slice['PPU'] - df_slice['PPU_hat'])*100/np.abs(df_slice['PPU'])
        df_track4 = pd.concat([df_track4, df_slice])
    df_track4 = df_track4.sort_values(by=['Manufacturer', 'Channel', 'Date'])
    df_track4 = df_track4[['Manufacturer', 'Channel', 'Date', 'PPU', 'PPU_hat', 'PPU MAPE',
                         'TDP', 'TDP_hat', 'TDP MAPE']]
    df_track4['forecast_month'] = (df_track4['Date'].min()- relativedelta(months = 1)).strftime("%B %Y")
    df_req_tdp_ppu = pd.concat([df_req_tdp_ppu, df_track4])
    df_track4 = df_track4.dropna(subset=['PPU'])
    df_res_ppu, df_res_tdp = tdp_ppu_table(df_track4, df_res_ppu, df_res_tdp)
    
    # Genearting engineered features and separating test timeframe data
    df_sc = df.copy()
    df_comp = generate_comp_features_1(df_sc)
    dates = sorted(df_comp['Date'].unique())[-1*n_days:]
    df_hold = df_comp[df_comp['Date'].isin(dates)]
    df_comp = df_comp[~df_comp['Date'].isin(dates)]

    # Future Data Preprocessing
    df_future = df_pred_whole[df_pred_whole['Date']>= min(dates)]
    df_future = df_future[['Date', 'Manufacturer', 'Channel', 'PPU_hat', 'TDP_hat', 'monthly_index_hat','quarterly_index_hat']]\
                        .rename(columns = {'PPU_hat': 'PPU',
                                           'TDP_hat': 'TDP',
                                           "monthly_index_hat":"monthly_index",
                                           "quarterly_index_hat":"quarterly_index"})
    df_future['TDP'] = df_future['TDP'].astype('int64')
    df_future_comp = generate_comp_features_1(df_future)

    # Appending and merging with actual data to get actual values for test timeframe 
    df_comp = pd.concat([df_comp, df_future_comp])

    df_comp = df_comp.merge(df_hold[['Date', 'Manufacturer', 'Channel', 'Units', 'Sales', 'SOM']]\
                                     .rename(columns = {'Units': 'Units_hold', 'Sales': 'Sales_hold', 'SOM': 'SOM_hold'}),
                             on= ['Date', 'Manufacturer', 'Channel'], how= 'left')

    for kpi in ['Units', 'Sales', 'SOM']:
        df_comp[kpi] = np.where(df_comp[kpi].isnull(), df_comp[f'{kpi}_hold'], df_comp[kpi])

    df_comp = df_comp.drop(columns = ['Units_hold', 'Sales_hold', 'SOM_hold'])
    df_comp = round(df_comp,2)

    # Fitting Best Sales model to get sales predictions
    test_months = n_days
    df_sales_pred, _, _, df_score = sales_prediction_constrained_regression(df_comp, future_months, test_months, target= 'Units')
    
    df_scores = pd.concat([df_scores,df_score])

    # Storing only future months predictions in a separate dataframe
    df_track = pd.DataFrame()
    # 7
    cols = ['Manufacturer', 'Channel', 'Date', 'SOM', 'SOM_hat', 'Sales', 'sales_pred', 'Units', 'units_pred']
    for keys, df_slice in df_sales_pred.groupby(['Channel', 'Manufacturer']):
        df_slice = df_slice.sort_values(by='Date')[cols].tail(future_months)
        df_track = pd.concat([df_track, df_slice])

    # Storing dataframes till actual data is available and generating performance matrix for Sales and SOM
    df_track2 = pd.DataFrame()
    for keys, df_slice in df_track.groupby(['Manufacturer', 'Channel']):
        # 6
        df_slice['units MAPE'] = np.abs(df_slice['Units'] - df_slice['units_pred'])*100/np.abs(df_slice['Units'])
        df_slice['sales MAPE'] = np.abs(df_slice['Sales'] - df_slice['sales_pred'])*100/np.abs(df_slice['Sales'])
        df_slice['SOM mae'] = np.abs(df_slice['SOM']-df_slice['SOM_hat'])
        df_track2 = pd.concat([df_track2, df_slice])
    df_track2 = df_track2.sort_values(by=['Manufacturer', 'Channel', 'Date'])
    # 5
    df_track2 = df_track2[['Manufacturer', 'Channel', 'Date', 'SOM', 'SOM_hat', 'SOM mae',
                         'Sales', 'sales_pred', 'sales MAPE', 'Units', 'units_pred', 'units MAPE']]
    df_track2['forecast_month'] = (df_track2['Date'].min()- relativedelta(months = 1)).strftime("%B %Y")
    df_req = pd.concat([df_req, df_track2])
    df_track2 = df_track2.dropna(subset=['Sales'])
# 4
    df_res_som, df_res_sales, df_res_units = som_sales_table(df_track2, df_res_som, df_res_sales, df_res_units)


# Rounding the values to 2 decimal places
df_res_ppu = round(df_res_ppu,2)
df_res_tdp = round(df_res_tdp,2)
df_req_tdp_ppu = round(df_req_tdp_ppu, 2)
df_req = round(df_req, 2)
df_res_som = round(df_res_som, 2)
df_res_sales = round(df_res_sales, 2)
# 3
df_res_units = round(df_res_units, 2)

# Renaming columns
df_res_ppu.rename(columns = {'training_till' : 'Training Upto', 'overall_wape' : 'Overall WAPE'}, inplace = True)
df_res_tdp.rename(columns = {'training_till' : 'Training Upto', 'overall_wape' : 'Overall WAPE'}, inplace = True)
df_req.rename(columns = {'sales_pred' : 'Sales_pred', 'sales MAPE' : 'Sales MAPE'}, inplace = True)
df_res_sales.rename(columns = {'training_till' : 'Training Upto', 'overall_wape' : 'Overall WAPE'}, inplace = True)
df_res_som.rename(columns = {'training_till' : 'Training Upto', 'overall_mae' : 'Overall MAE'}, inplace = True)
# 2
df_res_units.rename(columns = {'training_till' : 'Training Upto', 'overall_wape' : 'Overall WAPE'}, inplace = True)

# Saving the dataframes
df_res_ppu.to_csv(os.path.join(data_dir, ppu_error_file), index = False)
df_res_tdp.to_csv(os.path.join(data_dir, tdp_error_file), index = False)
df_req_tdp_ppu.to_csv(os.path.join(data_dir, ppu_tdp_req_file), index = False)
df_req.to_csv(os.path.join(data_dir, sales_req_file), index=False)
# 1
df_res_units.to_csv(os.path.join(data_dir, units_error_file), index=False)
df_res_sales.to_csv(os.path.join(data_dir, sales_error_file), index=False)
df_res_som.to_csv(os.path.join(data_dir, som_error_file), index=False)
df_scores.to_csv(os.path.join(data_dir,scores_filename), index=False)