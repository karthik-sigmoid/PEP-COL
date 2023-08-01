import io
import base64
import joblib
import math
import calendar
import numpy as np
import pandas as pd
from math import sqrt
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

import statsmodels.regression.linear_model as sm
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from scipy.optimize import lsq_linear
from itertools import chain, combinations

import dash_html_components as html
import dash_core_components as dcc
import dash_trich_components as dtc

from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, ElasticNet, SGDRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from prophet.forecaster import Prophet
from prophet.utilities import regressor_coefficients
from utilities.constants import *

def generate_regression_features(df):
    """ This function generates the required features 
        used in the model

    Args:
        df (dataframe)

    Returns:
        features: returns feature names as list
    """    
    avg_price_features = list()
    tdp_features = list()
    avg_price_ratio_features = list()
    tdp_ratio_features = list()

    mfgs = list(df['Manufacturer'].unique())
    features = ['PPU', 'TDP', 'PPU Ratio', 'TDP Ratio']

    for mfg in mfgs:
        for feat in features:
            feat_ = mfg + " " + feat
            if feat == 'PPU':
                avg_price_features.append(feat_)
            elif feat == 'TDP':
                tdp_features.append(feat_)
            elif feat == 'PPU Ratio':
                avg_price_ratio_features.append(feat_)
            else:
                tdp_ratio_features.append(feat_)

    return avg_price_features, tdp_features, avg_price_ratio_features, tdp_ratio_features


# Removing redundant features based on their statistical significance
def backward_elimination(data, target,significance_level = 0.05):
    """Eliminates excess variables,by deciding whether 
    the variable is useful for the model or not based on p values 

    Args:
        data (dataframe)
        target (dataframe): target variable 
        significance_level (float, optional): minimum significance value . Defaults to 0.05.

    Returns:
        features: list of features that are most relevant to a model
    """    
    features = data.columns.tolist()
    # while(len(features)>0):
    #     features_with_constant = add_constant(data[features])
    #     p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
    #     max_p_value = p_values.max()
    #     if(max_p_value >= significance_level):
    #         excluded_feature = p_values.idxmax()
    #         features.remove(excluded_feature)
    #     else:
    #         break
    return features


def MAPE(y_true, y_pred):
    """Calculates MAPE - Mean absolute percentage error

    Args:
        y_true : actual data
        y_pred : predicted data

    Returns:
        MAPE: single value (int/float)
    """    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true)))*100


def WAPE(y_true, y_pred):
    """Calculates WAPE

    Args:
        y_true : actual data
        y_pred : predicted data

    Returns:
        WAPE: single value (int/float)
    """    
    return (y_true - y_pred).abs().sum()*100 / y_true.abs().sum()


def sort_fun(x):
    """sort the given series according to giver order x

    Args:
        x (list): pre set order 

    Returns:
        list: list as per pre set order
    """    
    for i, thing in enumerate(sortList):
        if x.startswith(thing):
            return (i, x[len(thing):])


def generate_comp_features_1(data):
    """Generates competitor features 
    example: For PepsiCo it will get OTROS PPU and OTROS TDP

    Args:
        data (dataframe)

    Returns:
        dataframe: with competitor features
    """    
    df_comp = pd.DataFrame()
    for keys, df_slice in data.groupby(['Date', 'Channel']):
        df_t = df_slice.pivot( columns = 'Manufacturer', values = ['PPU'])
        df_t.columns =  [str(j + " " + i) for i, j in df_t.columns]
        df_t = df_t.fillna(0).sum()
        new_cols = df_t.index
        new_vals = df_t.values
        new_vals = df_t.values
        df_slice[new_cols] = new_vals

        df_t = df_slice.pivot( columns = 'Manufacturer', values = ['TDP'])
        df_t.columns =  [str(j + " " + i) for i, j in df_t.columns]
        df_t = df_t.fillna(0).sum()
        new_cols = df_t.index
        new_vals = df_t.values
        new_vals = df_t.values
        df_slice[new_cols] = new_vals
        df_comp = df_comp.append(df_slice)
    return df_comp


def generate_comp_features_2(data):
    """Generates competitor features 
    example: for PepsiCo it will get OTROS PPU/PEPSICO PPU
    and OTROS TDP/PEPSICO TDP

    Args:
        data (dataframe)

    Returns:
        dataframe: with competitor features
    """    
    data['Month'] = data['Date'].dt.month
    avg_cols = ['BIMBO PPU', 'DIANA PPU', 'DINANT PPU', 'KELLOGGS PPU', 'OTHERS PPU', 'PEPSICO PPU', 'SEÑORIAL PPU']
    tdp_cols = ['BIMBO TDP', 'DIANA TDP', 'DINANT TDP', 'KELLOGGS TDP', 'OTHERS TDP', 'PEPSICO TDP', 'SEÑORIAL TDP']

    for col in avg_cols:
        data[f'{col} Ratio'] = data[col].div(data.PPU)

    for col in tdp_cols:
        data[f'{col} Ratio'] = data[col].div(data.TDP)

    return data

def plot_dist_charts(pred_data, mfg):
    """plots yearly SOG charts

    Args:
        pred_data (dataframe): predicted dataframe
        mfg (str): manufacturer chosen from filter

    Returns:
        plot: yearly SOG graph
    """    
    pred_data = pred_data[['Date', 'Manufacturer', 'Channel', 'Sales', 'SOM', 'Predicted Sales', "Predicted SOM"]]
    pred_data['Date'] = pd.to_datetime(pred_data['Date'])
    pred_data['month'] = pred_data['Date'].dt.month
    pred_data['year'] = pred_data['Date'].dt.year
    min_year = pred_data['year'].min()
    pred_data = pred_data[pred_data['year']!=min_year]
    pred_data_1 = pred_data.groupby(['year', 'Channel', 'Manufacturer']).sum().reset_index()[['year',
                                                                                 'Channel',
                                                                                 'Manufacturer',
                                                                                 'Sales','Predicted Sales']]

    pred_data_1['sales_agg'] = pred_data_1.groupby('year')['Sales'].transform('sum')
    pred_data_1['SOG'] = pred_data_1['Sales']*100/pred_data_1['sales_agg']

    pred_data_1['pred_sales_agg'] = pred_data_1.groupby('year')['Predicted Sales'].transform('sum')
    pred_data_1['pred_SOG'] = pred_data_1['Predicted Sales']*100/pred_data_1['pred_sales_agg']

    pred_data_1 = round(pred_data_1,2)

    colors = {
                "A": '#0096D6',
                "B": '#00984A', 
                "C": '#EB7B30',
                "D": '#005CB4',
                "E": '#C9002B', 
                "F": '#00984A'
                }

    year_chart_df = pred_data_1.pivot(index = ['year', 'Manufacturer'],
                                      columns= 'Channel',
                                      values = ['SOG', 'pred_SOG']).reset_index()
    year = year_chart_df.year.max()

    # filter for manufacturer
    year_chart_df = year_chart_df[year_chart_df[('Manufacturer', '')] == mfg]
    last_sales_year = pred_data_1[pred_data_1['Sales']==0]['year'].min() - 1

    act_year_df = year_chart_df[year_chart_df[('year','')]<=last_sales_year]
    years = [str(i) for i in act_year_df.year.unique()]
    years[-1] = f'{str(last_sales_year)} YTD'
    act_year_df = act_year_df.fillna(0)
    act_year_df[('SOG', 'sum')] = act_year_df[('SOG',
                                        'OT')] + act_year_df[('SOG', 'TRADITIONAL')]
    
    plot = go.Figure(data=[go.Bar(
                            name = 'TRADITIONAL',
                            x = years,
                            y = act_year_df[('SOG','TRADITIONAL')],
                            text =  act_year_df[('SOG','TRADITIONAL')],
                            marker_color=colors['D'],
                            showlegend=False
                            ),
                            go.Bar(
                            name = 'OT',
                            x = years,
                            y = act_year_df[('SOG','OT')],
                            text = act_year_df[('SOG','OT')],
                            marker_color=colors['E'],
                            showlegend=False
                            ),
                            go.Scatter(
                            x = years,
                            y = act_year_df[('SOG','sum')],
                            text = round(act_year_df[('SOG','sum')],2),
                            mode = 'text',
                            textposition = 'top center',
                            textfont = dict(size =10),
                            showlegend=False
                           )])

    est_year_df = year_chart_df[year_chart_df[('year','')]>=last_sales_year]

    estimated_df_years = list(est_year_df[('year','')].unique())
    est_year_df = est_year_df.fillna(0)
    est_year_df[('pred_SOG', 'sum')] = est_year_df[('pred_SOG',
                                            'OT')] + est_year_df[('pred_SOG', 'TRADITIONAL')]

    annotation_height = np.nanmax([act_year_df[('SOG','sum')].max()+act_year_df[('SOG','sum')].max()*0.1, est_year_df[('pred_SOG','sum')].max()+est_year_df[('pred_SOG','sum')].max()*0.1])

    if len(act_year_df.year)>0:
            plot.add_annotation(x = len(act_year_df.year)/2, text = 'Actual',
                            y = annotation_height, showarrow = False)

    if estimated_df_years:
        plot.add_traces([go.Bar(
                            name = 'TRADITIONAL',
                            x = estimated_df_years,
                            y = est_year_df[('pred_SOG', 'TRADITIONAL')],
                            text =  est_year_df[('pred_SOG', 'TRADITIONAL')],
                            marker_color=colors['D']
                            ), 
                            go.Bar(
                            name = 'OT',
                            x = estimated_df_years,
                            y = est_year_df[('pred_SOG', 'OT')],
                            text = est_year_df[('pred_SOG', 'OT')],
                            marker_color=colors['E']
                            ),
                            go.Scatter(
                            x = estimated_df_years,
                            y = est_year_df[('pred_SOG','sum')],
                            text = round(est_year_df[('pred_SOG','sum')],2),
                            mode = 'text',
                            textposition = 'top center',
                            textfont = dict(size =10),
                            showlegend=False
                           )])

        plot.add_vline(x = len(act_year_df.year)-0.5, line_dash = 'dot')
        plot.add_annotation(x = len(act_year_df.year)-0.5+ (len(act_year_df.year) + len(est_year_df.year) - len(act_year_df.year))/2, text = 'Estimated',
                        y = annotation_height, showarrow = False)

    plot.update_layout(barmode='stack', title = f'{mfg} YoY SOG(Source of Growth) SOM',
                           paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9')
    return plot

def plot_monthly_dist_chart(pred_data, mfg, year):
    """plots monthly SOG charts

    Args:
        pred_data (dataframe): predicted dataframe
        mfg (str): manufacturer chosen from filter

    Returns:
        plot: monthly SOG graph
    """  
    pred_data_2 = pred_data.groupby(['Date', 'Channel', 'Manufacturer']).sum().reset_index()[['Date',
                                                                                 'Channel',
                                                                                 'Manufacturer',
                                                                                 'Sales','Predicted Sales']]

    colors = {
                "A": '#0096D6',
                "B": '#00984A',
                "C": '#EB7B30',
                "D": '#005CB4',
                "E": '#C9002B',
                "F": '#00984A'
                }

    pred_data_2['sales_agg'] = pred_data_2.groupby('Date')['Sales'].transform('sum')
    pred_data_2['SOG'] = pred_data_2['Sales']*100/pred_data_2['sales_agg']

    pred_data_2['pred_sales_agg'] = pred_data_2.groupby('Date')['Predicted Sales'].transform('sum')
    pred_data_2['pred_SOG'] = pred_data_2['Predicted Sales']*100/pred_data_2['pred_sales_agg']

    last_month_sales = pred_data_2[pred_data_2['Sales']!=0].Date.max()

    monthly_chart_df = pred_data_2.pivot(index=['Date','Manufacturer'], columns = 'Channel',
                                   values = ['SOG', 'pred_SOG']).reset_index()

    monthly_chart_df = round(monthly_chart_df, 2)
    monthly_chart_df['month'] = monthly_chart_df['Date'].dt.month
    monthly_chart_df['year'] = monthly_chart_df['Date'].dt.year

    monthly_chart_df['month_name'] = monthly_chart_df['month'].apply(lambda x:calendar.month_name[x])
    monthly_chart_df.drop("month", axis=1, inplace=True)
    monthly_chart_df.rename(columns= {"month_name":"month"}, inplace=True)

    monthly_chart_df = monthly_chart_df[(monthly_chart_df[('Manufacturer','')] == mfg) &
                                        (monthly_chart_df[('year','')] == year)
                                        ]

    # filter for manufacturer
    actual_df = monthly_chart_df[monthly_chart_df[('Date','')] <= last_month_sales]
    actual_df = actual_df.fillna(0)
    actual_df[('SOG',
              'sum')] = actual_df[('SOG', 
                                    'OT')] + actual_df[('SOG', 
                                                        'TRADITIONAL')]

    plot = go.Figure(data=[go.Bar(
                            name = 'TRADITIONAL',
                            x = actual_df.month,
                            y = actual_df[('SOG','TRADITIONAL')],
                            text =  actual_df[('SOG','TRADITIONAL')],
                            marker_color=colors['D'],
                            showlegend=False
                            ), 
                            go.Bar(
                            name = 'OT',
                            x = actual_df.month,
                            y = actual_df[('SOG','OT')],
                            text = actual_df[('SOG','OT')],
                            marker_color=colors['E'],
                            showlegend=False
                            ),
                            go.Scatter(
                            x = actual_df.month,
                            y = actual_df[('SOG','sum')],
                            text = round(actual_df[('SOG','sum')],2),
                            mode = 'text',
                            textposition = 'top center',
                            textfont = dict(size =10),
                            showlegend=False
                           )])

    estimated_df = monthly_chart_df[monthly_chart_df[('Date','')] > last_month_sales]
    estimated_df_months = list(estimated_df[('month','')].unique())
    estimated_df = estimated_df.fillna(0)
    estimated_df[('pred_SOG',
              'sum')] = estimated_df[('pred_SOG', 
                                    'OT')] + estimated_df[('pred_SOG', 
                                                        'TRADITIONAL')]

    annotation_height = np.nanmax([actual_df[('SOG','sum')].max()+actual_df[('SOG','sum')].max()*0.1, 
                                   estimated_df[('pred_SOG','sum')].max()+estimated_df[('pred_SOG','sum')].max()*0.1])
   
    if len(actual_df.month)>0:
        plot.add_annotation(x = len(actual_df.month)/2, text = 'Actual',
                        y = annotation_height, showarrow = False)

    if estimated_df_months:
        plot.add_traces([go.Bar(
                                name = 'TRADITIONAL',
                                x = estimated_df_months,
                                y = estimated_df[('pred_SOG', 'TRADITIONAL')],
                                text = estimated_df[('pred_SOG', 'TRADITIONAL')],
                                marker_color=colors['D']
                                ), 
                                go.Bar(
                                name = 'OT',
                                x = estimated_df_months,
                                y = estimated_df[('pred_SOG', 'OT')],
                                text = estimated_df[('pred_SOG', 'OT')],
                                marker_color=colors['E']
                                ),
                                go.Scatter(
                                x = estimated_df_months,
                                y = estimated_df[('pred_SOG','sum')],
                                text = round(estimated_df[('pred_SOG','sum')],2),
                                mode = 'text',
                                textposition = 'top center',
                                textfont = dict(size =10),
                                showlegend=False
                                )])

        plot.add_vline(x = len(actual_df.month)-0.5, line_dash = 'dot')
        plot.add_annotation(x = len(actual_df.month)-0.5+ (12 - len(actual_df.month))/2, text = 'Estimated',
                        y = annotation_height, showarrow = False)
    plot.update_layout(barmode='stack', title = f'{mfg} MoM SOG(Source of Growth) SOM for {year}',
                       paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9')
                      
    return plot


def diff_dashtable(data, data_previous, row_id_name=None):
    """Generates a diff of Dash DataTable data.

    Parameters
    ----------
    data: DataTable property (https://dash.plot.ly/datatable/reference)
        The contents of the table (list of dicts)
    data_previous: DataTable property
        The previous state of `data` (list of dicts).

    Returns
    -------
    A list of dictionaries in form of [{row_index:, column_id:, current_value:,
        previous_value:}]
    """
    df, df_previous = pd.DataFrame(data=data), pd.DataFrame(data_previous)

    if row_id_name is not None:
        # If using something other than the index for row id's, set it here
        for _df in [df, df_previous]:

            # Why do this?  Guess just to be sure?
            assert row_id_name in _df.columns

            _df = _df.set_index(row_id_name)
    else:
        row_id_name = "row_index"

    # Mask of elements that have changed, as a dataframe.  Each element indicates True if df!=df_prev
    df_mask = ~((df == df_previous) | ((df != df) & (df_previous != df_previous)))

    # ...and keep only rows that include a changed value
    df_mask = df_mask.loc[df_mask.any(axis=1)]

    changes = []

    # This feels like a place I could speed this up if needed
    for idx, row in df_mask.iterrows():
        row_id = row.name

        # Act only on columns that had a change
        row = row[row.eq(True)]

        for change in row.iteritems():

            changes.append(
                {
                    row_id_name: row_id,
                    "column_id": change[0],
                }
            )

    return changes


def get_sales_kpi_values(df):
    """generates Sales Mape and SOM MAE for past 1, 6, and 12 months data 

    Args:
        df (dataframe)

    Returns:
        dictionary: results = {'PEPSICO_DTS': [six_months_wape, six_months_mae, twelve_months_wape, twelve_months_mae], ...}
    """    
    cols = ['Manufacturer','Channel','Date','Units','Sales','Units_pred', 'Sales_pred','SOM mae','forecast_month']
    df_req = pd.DataFrame(columns = cols)
    for keys, df_slice in df.groupby(['Manufacturer','Channel','forecast_month']):
        mfr, cnl, trained_till = keys
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        df_slice = df_slice.sort_values(by='Date', ascending = True)
        df_req.loc[len(df_req)] = df_slice[cols].head(1).values[0]
    df_req = df_req.sort_values(by = ['Manufacturer','Channel','Date'], ascending=True)

    results = dict()

    for keys, df_slice in df_req.groupby(['Manufacturer','Channel']):
        mfr, cnl = keys
        df_slice = df_slice.sort_values(by='Date', ascending=True)
        twelve_months_mae = round(df_slice['SOM mae'].mean(),2)
        twelve_months_wape = round(WAPE(df_slice['Sales'], df_slice['Sales_pred']),2)
        twelve_months_units_wape = round(WAPE(df_slice['Units'], df_slice['Units_pred']),2)

        six_months_df = df_slice.tail(5)
        six_months_mae = round(six_months_df['SOM mae'].mean(),2)
        six_months_wape = round(WAPE(six_months_df['Sales'], df_slice['Sales_pred']),2)
        six_months_units_wape = round(WAPE(six_months_df['Units'], df_slice['Units_pred']),2)

        results[f'{mfr}_{cnl}'] = [six_months_units_wape, six_months_wape, six_months_mae, 
                                   twelve_months_units_wape, twelve_months_wape, twelve_months_mae]
    return results

def get_kpi_values(df, results, col = 'PPU'):
    """generates PPU/TDP Mape for past 1, 6, and 12 months data 

    Args:
        df (dataframe)
        results (dictionary): results = {'PEPSICO_DTS': [six_months_wape, six_months_mae, twelve_months_wape, twelve_months_mae], ...}
        col (str, optional): string. Defaults to 'PPU'.

    Returns:
       dictionary: results = {'PEPSICO_DTS': [six_months_wape, six_months_mae, twelve_months_wape, twelve_months_mae, 
                                              six_months_wape, twelve_months_wape], ...}

    """    
    cols = ['Manufacturer','Channel','Date',col,f'{col}_hat','forecast_month']
    df_req = pd.DataFrame(columns = cols)
    for keys, df_slice in df.groupby(['Manufacturer','Channel','forecast_month']):
        mfr, cnl, trained_till = keys
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        df_slice = df_slice.sort_values(by='Date', ascending = True)
        df_req.loc[len(df_req)] = df_slice[cols].head(1).values[0]
    df_req = df_req.sort_values(by = ['Manufacturer','Channel','Date'], ascending=True)


    for keys, df_slice in df_req.groupby(['Manufacturer','Channel']):
        mfr, cnl = keys
        df_slice = df_slice.sort_values(by='Date', ascending=True)
        twelve_months_wape = round(WAPE(df_slice[col], df_slice[f'{col}_hat']),2)

        six_months_df = df_slice.tail(5)
        six_months_wape = round(WAPE(six_months_df[col], df_slice[f'{col}_hat']),2)

        results[f'{mfr}_{cnl}'] = results[f'{mfr}_{cnl}']+[six_months_wape, twelve_months_wape]

    return results


def preprocessing_future_data(filename):
    """converts excel file with different tabs into a readable dataframe 

    Args:
        filename

    Returns:
        dataframe: readable dataframe for models
    """    

    if 'xlsb' in filename:
        engine = 'pyxlsb'
    else:
        engine = 'openpyxl'
    df_agg = pd.DataFrame()
    sheets_dict = pd.read_excel(filename, engine= engine, sheet_name=None, header= None)
    # print(list(sheets_dict.keys()))
    for sheetname in sheets_dict.keys():
        channel = sheetname.strip().upper()
        df = sheets_dict[sheetname].copy()
        df = df.dropna(thresh = 5).dropna(axis = 1, how='all').dropna(how='all')
        df.columns = df.iloc[0]

        change_cols = []
        for i in df.columns:
            if (str(i).startswith("Unnamed") or i is np.nan):
                change_cols.append(np.nan)
            else:
                change_cols.append(i)

        df.columns = change_cols

        df.columns = df.columns.to_series().mask(lambda x: x==np.nan).ffill()

        df = df.iloc[1:]
        df = df.T.reset_index()
        month_cols = [x for x in df.iloc[0].values if (str(x) != 'nan') and (str(x) != 'Periodo') and (str(x).startswith('Unnamed')==False)]

        df.columns = df.iloc[0]
        df.columns = ['Manufacturer', 'cols'] + month_cols

        df = df.iloc[1:]
        df = pd.melt(df, id_vars=['Manufacturer', 'cols'], value_vars= month_cols, var_name= 'Date')
        df = df.pivot(index=['Manufacturer', 'Date'], columns='cols').reset_index()
        df.columns = [i[0] if i[1]=="" else i[1] for i in df.columns]
        df = df.rename(columns= {'TDPs': 'TDP'})
        df['Manufacturer'] = df['Manufacturer'].str.upper()
        df['date_new'] = df['Date'].replace(month_dict, regex=True)
        df['date_new'] = pd.to_datetime(df['date_new'], format='%B %Y')
        df['Channel'] = channel
        df_agg = pd.concat([df_agg, df])
    df_agg = df_agg.drop(columns= ['Date'])
    df_agg = df_agg.rename(columns= {'date_new': 'Date'})

    return df_agg


def process_features_file(dataframe):
    """converts model features file into a readable format (
        specifically while reading file as csv, everything is in string format,
        this function converts the list of features from string format to 
        list format
    )

    Args:
        dataframe: model features data

    Returns:
        dataframe
    """    

    dataframe['features'] = dataframe['features'].apply(lambda x : x.strip("["))
    dataframe['features'] = dataframe['features'].apply(lambda x : x.strip("]"))
    dataframe['features'] = dataframe['features'].apply(lambda x : x.replace("'",""))
    dataframe['features'] = dataframe['features'].apply(lambda x : [i.lstrip(" ") for i in x.split(",")])
    # dataframe['features'] = dataframe['features'].apply(lambda x : [i.upper() for i in x])
    # dataframe['features'] = dataframe['features'].apply(lambda x : [i.replace('AVG_PRICE','PPU') for i in x])
    # dataframe['features'] = dataframe['features'].apply(lambda x : [i.replace('RATIO','Ratio') for i in x])

    return dataframe

def predict_sales(dataframe, baseline_prediction_data, df_coef, model_features):
    """predicts sales based on the uploaded values in scenario management page

    Args:
        dataframe
        baseline_prediction_data: predictions based on baseline model
        model_features (list): list of features based on channel and manufacturer

    Returns:
        dataframe : dataframe with sales prediction based on the uploaded values
    """    
    global regression_features, ext_features
    df_sales_whole = pd.DataFrame()

    for keys, df_slice in dataframe.groupby(['Manufacturer','Channel']):
        mfr, cnl = keys
        req_data = model_features[model_features['combination'] == f"{cnl}_{mfr}"].values
        assert req_data.shape[0] == 1
        combination = req_data[0][0]
        model_ = req_data[0][1]
        features = req_data[0][2]

        features_ = list()
        for feat in features:
            if feat.isnumeric():
                features_.append(int(feat))
            else:
                features_.append(feat)

        # print(features_)
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        df_slice['PPU'] = df_slice[f"{mfr} PPU"]
        df_slice['TDP'] = df_slice[f"{mfr} TDP"]

        # data_df = generate_comp_features_2(df_slice)
        data_df = df_slice.copy()
        data_df = data_df.drop(['PPU','TDP'], axis=1)
        data_df = data_df.dropna(axis=1, how='all')
        data_df['Month'] = data_df['Date'].dt.month
        # print(data_df.head())

        data_df = data_df.merge(baseline_prediction_data[['Date',
                                            'Channel', 
                                            'Manufacturer']+seasonality_cols].drop_duplicates(), 
                                            on=['Date',
                                            'Channel', 
                                            'Manufacturer'], how='left')
        
        # df_slice = df_slice.merge(baseline_prediction_data[['Date',
        #                                             'Channel', 
        #                                             'Manufacturer']+ext_features].drop_duplicates(), 
        #                                             on=['Date',
        #                                             'Channel', 
        #                                             'Manufacturer'], how='left')
        sel_feat = features_
        log_features = [i for i in sel_feat]
        
        oh_cols = list(pd.get_dummies(data_df['Month']).columns.values)
        encoded_features = pd.get_dummies(data_df['Month'])
        data_df = pd.concat([data_df, encoded_features],axis=1)
        data_df = data_df.sort_values(by='Date')
        data_df[log_features] = np.where(data_df[log_features] > 0, data_df[log_features].astype(float).apply(np.log10), 
                                         np.where(data_df[log_features] < 0, data_df[log_features], 0))
        coeff = df_coef[(df_coef['Manufacturer']==mfr)&(df_coef['Channel']==cnl)]
        y = final_processing(data_df, sel_feat, coeff)
        mean_value = np.mean(y[-18:])
        df_slice['User modified Units Prediction'] = y.values
        df_slice['User modified Units Prediction'] = np.where(df_slice['User modified Units Prediction'] < 0,
                                                               mean_value, df_slice['User modified Units Prediction'])
        
        # print(df_slice.tail())
        df_sales_whole = pd.concat([df_sales_whole, df_slice])
    # print(df_sales_whole.tail())
    df_sales_whole['User modified Sales Prediction'] = df_sales_whole['User modified Units Prediction'] * df_sales_whole['PPU'].astype(float)
    df_sales_whole['total_sales_hat'] = df_sales_whole.groupby(['Date',
                                            'Channel'])['User modified Sales Prediction'].transform('sum')

    df_sales_whole['User modified SOM Prediction'] = df_sales_whole['User modified Sales Prediction']*100/df_sales_whole['total_sales_hat']
    # print(df_sales_whole.tail())

    df_sales_whole = df_sales_whole.drop('total_sales_hat', axis=1)

    baseline_prediction_data.rename(columns = {"units_pred":'Baseline Units Prediction',
                                                "sales_pred":'Baseline Sales Prediction',
                                                "SOM_hat":'Baseline SOM Prediction'},
                                                inplace=True)

    df_sales_whole = df_sales_whole.merge(baseline_prediction_data[['Date','Manufacturer',
                                            'Channel','Baseline Units Prediction', 
                                            'Baseline Sales Prediction',
                                            'Baseline SOM Prediction']],
                                            on = ['Date','Manufacturer','Channel'])
    
    for col in ['Baseline Units Prediction', 'Baseline Sales Prediction', 'User modified Units Prediction', 'User modified Sales Prediction']:
        df_sales_whole[col] = df_sales_whole[col].astype(float)
    df_sales_all = df_sales_whole.groupby(['Manufacturer', 'Date'])[['Baseline Units Prediction', 'Baseline Sales Prediction', 'User modified Units Prediction', 'User modified Sales Prediction']].sum().reset_index()
    df_sales_all['Channel'] = "TOTAL"
    
    df_sales_all['total_sales'] = df_sales_all.groupby(['Date', 
                                                        'Channel'])['Baseline Sales Prediction'].transform('sum')
    df_sales_all['Baseline SOM Prediction'] = df_sales_all['Baseline Sales Prediction']*100/df_sales_all['total_sales']
    del df_sales_all['total_sales']
    
    df_sales_all['total_sales_hat'] = df_sales_all.groupby(['Date', 
                                                            'Channel'])['User modified Sales Prediction'].transform('sum')
    
    df_sales_all['User modified SOM Prediction'] = df_sales_all['User modified Sales Prediction']*100/df_sales_all['total_sales_hat']
    del df_sales_all['total_sales_hat']
    df_sales_all['Month'] = df_sales_all['Date'].dt.month

    df_sales_whole = pd.concat([df_sales_whole, df_sales_all])

    return df_sales_whole



def save_double_column_df(df, xl_writer, startrow = 0, **kwargs):
    '''Function to save doublecolumn DataFrame, to xlwriter'''
    # inputs:
    # df - pandas dataframe to save
    # xl_writer - book for saving
    # startrow - row from wich data frame will begins
    # **kwargs - arguments of `to_excel` function of DataFrame`
    df.drop(df.index).to_excel(xl_writer, startrow = startrow, **kwargs)
    df.to_excel(xl_writer, startrow = startrow + 1, header = False, **kwargs)


def get_channel_df(dataframe, channel):
    df_comb = dataframe[(dataframe['Channel']==channel)]
    df_comb = df_comb.dropna(axis=1, how='all')
    df_comb = df_comb.sort_values(by= 'Date')
    df_comb = df_comb.drop(columns= ['Manufacturer', 'Channel', 'year', 'Date']).set_index('key')
    df_comb = df_comb.drop_duplicates()
    df_comb = df_comb.T
    df_comb = df_comb.reindex(sortList)
    df_comb = df_comb.dropna(how='all')

    df_comb.index = pd.MultiIndex.from_tuples([tuple(i.rsplit(' ', 1)) for i in df_comb.index])
    df_comb = df_comb.T

    df_comb.index.name = None

    return df_comb


# def sales_prediction_best_model(data, future_months, test_months, df_feat_imp, target, save_result=True):
#     """selects the best possible model based on a metric, 
#     saves the model into model directory, 
#     predicts future data 

#     Args:
#         data (dataframe)
#         future_months (int): no of future months we want to predict
#         test_months (int): no of months to be used as testing data in the existing data to select the best model
#         df_feat_imp (dataframe): feature importance dataframe (feature importance is being calculated and saved)
#         save_result (bool, optional): whether to save the above files or not (used differently at different stages of code). Defaults to True.

#     Returns:
#         tuple of dataframe: df_sales_pred --> sales prediction dataframe, 
#         model_selected_feat --> modelname, channel and manufacturer and selected features are saved ,
#         df_feat_imp --> feature importance data is saved
#         df_scores --> dataframe with trained model tracker for train test sets
#     """    
#     global regression_features

#     wape_dict = {}
#     model_selected_feat = pd.DataFrame(columns= ['combination', 'model', 'features'])
#     df_sales_pred = pd.DataFrame()
#     df_scores = pd.DataFrame(columns = ['Month','Channel','Manufacturer','Model Selected', 
#                                         'Training Period',f'Training Error ({target} WAPE in %)',
#                                         "Testing Period",f'Testing Error ({target} WAPE in %)'])


#     for keys, df_slice in data.groupby(['Manufacturer', 'Channel']):
#         print("Running best sales model for:",keys)
#         mfr, cnl = keys
#         df_slice = df_slice.sort_values(by='Date')
#         df_slice[regression_features + ext_features] = df_slice[regression_features + ext_features].fillna(0)

#         oh_cols = list(pd.get_dummies(df_slice['Month']).columns.values)[1:]
#         encoded_features = pd.get_dummies(df_slice['Month'])
#         df_slice2 = pd.concat([df_slice, encoded_features],axis=1)
#         df_slice3 = df_slice2.sort_values(by='Date').iloc[:-1*future_months]
#         min_date = df_slice3.Date.min()
#         max_date = df_slice3.Date.max()
#         train_period = min_date.strftime("%b %y") + " - " +(max_date - relativedelta(months=4)).strftime("%b %y")
#         test_period = (max_date - relativedelta(months=3)).strftime("%b %y") + " - " + max_date.strftime("%b %y")
#         X, y = df_slice3[regression_features + oh_cols], df_slice3[[target]]
        
#         X[regression_features] = np.where(X[regression_features] != 0, X[regression_features].apply(np.log10), 0)

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_months, shuffle = False)

#         X_train = X_train.dropna(how='all')
#         X_test = X_test.dropna(how='all')

#         # Feature selection for regression based models
#         selected_features = backward_elimination(X_train, y_train)

#         X_train, X_test = X_train[selected_features], X_test[selected_features]

#         # Linear Regression
#         model = LinearRegression()
#         model.fit(X_train, y_train)
#         y_train_pred = model.predict(X_train)
#         y_test_pred = model.predict(X_test)
#         linear_train_wape = WAPE(y_train[target], y_train_pred.reshape(-1).tolist())
#         linear_test_wape = WAPE(y_test[target], y_test_pred.reshape(-1).tolist())

#         # Ridge Regression
#         grid = dict()
#         grid['alpha'] = np.arange(0, 1, 0.01)
#         model = Ridge()
#         search_ridge = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
#         results_ridge = search_ridge.fit(X_train, y_train)
#         model = Ridge(**results_ridge.best_params_)
#         model.fit(X_train, y_train)
#         y_train_pred = model.predict(X_train)
#         y_test_pred = model.predict(X_test)
#         ridge_train_wape = WAPE(y_train[target], y_train_pred.reshape(-1).tolist())
#         ridge_test_wape = WAPE(y_test[target], y_test_pred.reshape(-1).tolist())

#         # Lasso Regression Tuned
#         model = Lasso()
#         search_lasso = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
#         results_lasso = search_lasso.fit(X_train, y_train)
#         model = Lasso(**results_lasso.best_params_)
#         model.fit(X_train, y_train)
#         y_train_pred = model.predict(X_train)
#         y_test_pred = model.predict(X_test)
#         lasso_train_wape = WAPE(y_train[target], y_train_pred.reshape(-1).tolist())
#         lasso_test_wape = WAPE(y_test[target], y_test_pred.reshape(-1).tolist())

#         # Bayesian Regression
#         model = BayesianRidge()
#         model.fit(X_train, y_train)
#         y_train_pred = model.predict(X_train)
#         y_test_pred = model.predict(X_test)
#         bayesian_train_wape = WAPE(y_train[target], y_train_pred.reshape(-1).tolist())
#         bayesian_test_wape = WAPE(y_test[target], y_test_pred.reshape(-1).tolist())

#         # ElasticNet Regression Tuned
#         grid = {}
#         grid['l1_ratio'] = np.arange(0, 1, 0.01)
#         grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
#         model = ElasticNet()
#         search_elastic = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
#         results_elastic = search_elastic.fit(X_train, y_train)
#         model = ElasticNet(**results_elastic.best_params_)
#         model.fit(X_train, y_train)
#         y_train_pred = model.predict(X_train)
#         y_test_pred = model.predict(X_test)
#         elastic_train_wape = WAPE(y_train[target], y_train_pred.reshape(-1).tolist())
#         elastic_test_wape = WAPE(y_test[target], y_test_pred.reshape(-1).tolist())

#         # SGD Regressor
#         model = SGDRegressor(random_state=20)
#         model.fit(X_train, y_train)
#         y_train_pred = model.predict(X_train)
#         y_test_pred = model.predict(X_test)
#         sgd_train_wape = WAPE(y_train[target], y_train_pred.reshape(-1).tolist())
#         sgd_test_wape = WAPE(y_test[target], y_test_pred.reshape(-1).tolist())

#         # Prophet
#         df1 = df_slice[['Date', target]+regression_features].rename(columns={target:'y', 'Date':'ds'})
#         df1 = df1.sort_values(by='ds').reset_index(drop=True).iloc[:-1*future_months]
#         train, test = df1.iloc[:-1*test_months], df1.iloc[-1*test_months:]

#         X_prophet, y_prophet = train[regression_features], train[['y']]
#         X_prophet[regression_features] = np.where(X_prophet[regression_features]>0,
#                                                   X_prophet[regression_features].apply(np.log10), 0)
#         train[regression_features] = np.where(train[regression_features]>0,
#                                               train[regression_features].apply(np.log10), 0)
#         test[regression_features] = np.where(test[regression_features]>0,
#                                              test[regression_features].apply(np.log10), 0)
#         selected_features = backward_elimination(X_prophet, y_prophet)

#         model = Prophet(daily_seasonality= False,
#                     weekly_seasonality= False,
#                     yearly_seasonality=False
#                        ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
#                              .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)

#         for x in selected_features:
#             model.add_regressor(x)

#         model.fit(train)
#         forecast_train = model.predict(train[['ds']+selected_features]).merge(train[['ds', 'y']], on= ['ds'])
#         forecast_test = model.predict(test[['ds']+selected_features]).merge(test[['ds', 'y']], on= ['ds'])

#         prophet_train_wape =  WAPE(forecast_train['y'], forecast_train['yhat'])
#         prophet_test_wape =  WAPE(forecast_test['y'], forecast_test['yhat'])

#         # Selecting model having minimum test wape
#         wape_dict['wape_linear'] = linear_test_wape
#         wape_dict['wape_ridge'] = ridge_test_wape
#         wape_dict['wape_lasso'] = lasso_test_wape
#         # wape_dict['wape_bayesian'] = bayesian_test_wape
#         # wape_dict['wape_elastic'] = elastic_test_wape
#         # wape_dict['wape_sgd'] = sgd_test_wape
#         wape_dict['wape_prophet'] = prophet_test_wape


#         model_select = min(wape_dict, key = wape_dict.get)

#         if model_select == 'wape_linear':
#             selected_features = backward_elimination(X, y)
#             model_selected_feat.loc[len(model_selected_feat)] = [cnl + "_" + mfr, "Linear", selected_features]
#             X = X[selected_features]
#             model = LinearRegression()
#             model.fit(X, y)

#             filename = f'Linear_{cnl}_{mfr}.sav'

#             df_scores.loc[len(df_scores)] = [max_date.strftime("%b-%y"), cnl, mfr, model_select.split("_")[-1].title(), 
#                                              train_period, linear_train_wape, test_period, linear_test_wape]

#             # Feature importance
#             weights = model.coef_[0]
#             df_feat_imp = feature_importance(mfr, cnl, selected_features, X, weights, df_feat_imp)
#             df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0,
#                                                       df_slice2[regression_features].apply(np.log10), 0)
#             df_slice['units_pred'] = model.predict(df_slice2[selected_features])
#             df_sales_pred = pd.concat([df_sales_pred, df_slice])

#         elif model_select == 'wape_ridge':
#             selected_features = backward_elimination(X, y)
#             model_selected_feat.loc[len(model_selected_feat)] = [cnl + "_" + mfr, "Ridge", selected_features]
#             X = X[selected_features]

#             model = Ridge(**results_ridge.best_params_)
#             model.fit(X, y)

#             filename = f'Ridge_{cnl}_{mfr}.sav'
#             df_scores.loc[len(df_scores)] = [max_date.strftime("%b-%y"), cnl, mfr, model_select.split("_")[-1].title(), 
#                                              train_period, ridge_train_wape, test_period, ridge_test_wape]

#             # Feature importance
#             weights = model.coef_[0]
#             df_feat_imp = feature_importance(mfr, cnl, selected_features, X, weights, df_feat_imp)
#             df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0,
#                                                       df_slice2[regression_features].apply(np.log10), 0)
#             df_slice['units_pred'] = model.predict(df_slice2[selected_features])
#             df_sales_pred = pd.concat([df_sales_pred, df_slice])

#         elif model_select == 'wape_lasso':
#             selected_features = backward_elimination(X, y)
#             model_selected_feat.loc[len(model_selected_feat)] = [cnl + "_" + mfr, "Lasso", selected_features]
#             X = X[selected_features]

#             model = Lasso(**results_lasso.best_params_)
#             model.fit(X, y)

#             filename = f'Lasso_{cnl}_{mfr}.sav'
#             df_scores.loc[len(df_scores)] = [max_date.strftime("%b-%y"), cnl, mfr, model_select.split("_")[-1].title(), 
#                                              train_period, lasso_train_wape, test_period, lasso_test_wape]

#             # Feature importance
#             weights = model.coef_[0]
#             df_feat_imp = feature_importance(mfr, cnl, selected_features, X, weights, df_feat_imp)

#             df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0,
#                                                       df_slice2[regression_features].apply(np.log10), 0)
#             df_slice['units_pred'] = model.predict(df_slice2[selected_features])
#             df_sales_pred = pd.concat([df_sales_pred, df_slice])


#         elif model_select == 'wape_bayesian':
#             selected_features = backward_elimination(X, y)
#             model_selected_feat.loc[len(model_selected_feat)] = [cnl + "_" + mfr, "Bayesian", selected_features]
#             X = X[selected_features]

#             model = BayesianRidge()
#             model.fit(X, y)
#             filename = f'Bayesian_{cnl}_{mfr}.sav'
#             df_scores.loc[len(df_scores)] = [max_date.strftime("%b-%y"), cnl, mfr, model_select.split("_")[-1].title(), 
#                                              train_period, bayesian_train_wape, test_period, bayesian_test_wape]

#             # Feature importance
#             weights = model.coef_[0]
#             df_feat_imp = feature_importance(mfr, cnl, selected_features, X, weights, df_feat_imp)

#             df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0,
#                                                       df_slice2[regression_features].apply(np.log10), 0)
#             df_slice['units_pred'] = model.predict(df_slice2[selected_features])
#             df_sales_pred = pd.concat([df_sales_pred, df_slice])


#         elif model_select == 'wape_elastic':
#             selected_features = backward_elimination(X, y)
#             model_selected_feat.loc[len(model_selected_feat)]=[cnl + "_" + mfr, "ElasticNet", selected_features]
#             X = X[selected_features]

#             model = ElasticNet(**results_elastic.best_params_)
#             model.fit(X, y)
#             filename = f'ElasticNet_{cnl}_{mfr}.sav'
#             df_scores.loc[len(df_scores)] = [max_date.strftime("%b-%y"), cnl, mfr, model_select.split("_")[-1].title(), 
#                                              train_period, elastic_train_wape, test_period, elastic_test_wape]

#             # Feature importance
#             weights = model.coef_[0]
#             df_feat_imp = feature_importance(mfr, cnl, selected_features, X, weights, df_feat_imp)

#             df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0,
#                                                       df_slice2[regression_features].apply(np.log10), 0)
#             df_slice['units_pred'] = model.predict(df_slice2[selected_features])
#             df_sales_pred = pd.concat([df_sales_pred, df_slice])

#         elif model_select == 'wape_sgd':
#             selected_features = backward_elimination(X, y)
#             model_selected_feat.loc[len(model_selected_feat)] = [cnl + "_" + mfr, "SGD", selected_features]
#             X = X[selected_features]

#             model = SGDRegressor(random_state=20)
#             model.fit(X, y)
#             filename = f'SGD_{cnl}_{mfr}.sav'
#             df_scores.loc[len(df_scores)] = [max_date.strftime("%b-%y"), cnl, mfr, model_select.split("_")[-1].title(), 
#                                              train_period, sgd_train_wape, test_period, sgd_test_wape]

#             # Feature importance
#             weights = model.coef_[0]
#             df_feat_imp = feature_importance(mfr, cnl, selected_features, X, weights, df_feat_imp)

#             df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0,
#                                                       df_slice2[regression_features].apply(np.log10), 0)
#             df_slice['units_pred'] = model.predict(df_slice2[selected_features])
#             df_sales_pred = pd.concat([df_sales_pred, df_slice])

#         elif model_select == 'wape_prophet':
#             df1 = df_slice[['Date', target]+regression_features].rename(columns={target:'y', 'Date':'ds'})
#             train = pd.concat([train, test])

#             X, y = train[regression_features], train[['y']]
#             selected_features = backward_elimination(X, y)
#             model_selected_feat.loc[len(model_selected_feat)] = [cnl + "_" + mfr, "Prophet", selected_features]

#             model = Prophet(daily_seasonality= False,
#                     weekly_seasonality= False,
#                     yearly_seasonality=False,
#                        ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
#                              .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)
#             for x in selected_features:
#                 model.add_regressor(x)

#             model.fit(train)
#             filename = f'Prophet_{cnl}_{mfr}.sav'
#             df_scores.loc[len(df_scores)] = [max_date.strftime("%b-%y"), cnl, mfr, model_select.split("_")[-1].title(), 
#                                              train_period, prophet_train_wape, test_period, prophet_test_wape]

#             # Feature importance
#             if len(selected_features)>0:
#                 regressor_coef = regressor_coefficients(model)
#                 weights = regressor_coef['coef'].values
#                 df_feat_imp = feature_importance(mfr, cnl, selected_features, X, weights, df_feat_imp)
#             else:
#                 importance = pd.DataFrame()
#                 df_feat_imp = pd.concat([df_feat_imp, importance])

#             df1[regression_features] = np.where(df1[regression_features] != 0,
#                                                       df1[regression_features].apply(np.log10), 0)
#             df_slice = df_slice.sort_values(by= 'Date')
#             df_slice['units_pred'] = model.predict(df1[selected_features + ['ds']])['yhat'].values
#             df_sales_pred = pd.concat([df_sales_pred, df_slice])

#         if save_result:
#             model_selected_feat.to_csv(os.path.join(data_dir,selected_features_filename), index=False)
#             df_feat_imp.to_csv(os.path.join(data_dir,feature_importance_filename), index=False)

#             if os.path.exists(model_dir):
#                 joblib.dump(model, os.path.join(model_dir,filename))
#             else:
#                 os.makedirs(model_dir)
#                 joblib.dump(model, os.path.join(model_dir,filename))

      

#     # SOM Calculation
#     df_sales_pred['units_pred'] = np.where(df_sales_pred['units_pred']<0, 0, df_sales_pred['units_pred'])
#     df_sales_pred['sales_pred'] = df_sales_pred['units_pred'] * df_sales_pred['PPU']
#     df_sales_pred['total_sales_hat'] = df_sales_pred.groupby(['Date', 'Channel'])['sales_pred'].transform('sum')
#     df_sales_pred['SOM_hat'] = df_sales_pred['sales_pred']*100/df_sales_pred['total_sales_hat']
#     del df_sales_pred['total_sales_hat']
    
#     df_scores = df_scores.round(decimals = 2)

#     return df_sales_pred, model_selected_feat, df_feat_imp, df_scores


def feature_importance(df_coef):
    """generates data for plotting feature importance chart in Baseline Prediction page

    Args:
        df_coef (dataframe)
    Returns:
        dataframe : df_feat_imp
    """
    df_feat_imp = df_coef[df_coef['features']!= 'intercept']
    df_feat_imp = df_feat_imp[~df_feat_imp['features'].isin(seasonality_cols)]
    df_feat_imp['feature_importance_vals'] = abs(df_feat_imp['coef'])
    df_feat_imp['feature_importance_vals'] = df_feat_imp.groupby(['Manufacturer', 'Channel'])\
                                                ['feature_importance_vals'].transform('sum')
    df_feat_imp['feature_importance_vals'] = df_feat_imp['coef']/df_feat_imp['feature_importance_vals']
    df_feat_imp = df_feat_imp[(df_feat_imp['feature_importance_vals']>0.001)|
                                      (df_feat_imp['feature_importance_vals']<-0.001)]
    df_feat_imp = df_feat_imp.sort_values(by = 'feature_importance_vals',  ascending= True)
    df_feat_imp["Color"] = np.where(df_feat_imp["feature_importance_vals"]<0, '#C9002B', '#005CB4')
    df_feat_imp = df_feat_imp.rename(columns= {'features': 'Feature'})
    df_feat_imp = df_feat_imp[['Feature', 'feature_importance_vals', 'Color', 'Manufacturer', 'Channel']]
    return df_feat_imp


def som_sales_table(df_track2, df_res_som, df_res_sales, df_res_units):
    """generates sales and som error metrics table used in prediction accuracy page

    Args:
        df_track2 (dataframe)
        df_res_som (dataframe)
        df_res_sales (dataframe)
    Returns:
        tuple of dataframes : df_res_som, df_res_sales
    """    

    training_till = (df_track2['Date'].min()- relativedelta(months = 1)).strftime("%B %Y")
    for keys, df_slice in df_track2.groupby(['Manufacturer', 'Channel']):
        df_slice['training_till'] = training_till
        df_slice = pd.pivot(df_slice, index= ['Manufacturer', 'Channel', 'training_till'], columns= ['Date'],
                          values= ['SOM mae'])
        df_slice.columns = [pd.to_datetime(m[1]).strftime("%B %Y") for m in df_slice.columns]
        df_slice = df_slice.reset_index()
        df_slice['overall_mae'] = df_slice.mean(axis=1)
        df_res_som = pd.concat([df_res_som, df_slice])

    for keys, df_slice in df_track2.groupby(['Manufacturer', 'Channel']):
        df_slice['training_till'] = training_till
        df_slice2 = pd.pivot(df_slice, index= ['Manufacturer', 'Channel', 'training_till'], columns= ['Date'],
                          values= ['sales MAPE'])
        df_slice2.columns = [pd.to_datetime(m[1]).strftime("%B %Y") for m in df_slice2.columns]
        df_slice2 = df_slice2.reset_index()
        df_slice2['overall_wape'] = WAPE(df_slice['Sales'], df_slice['sales_pred'])
        df_res_sales = pd.concat([df_res_sales, df_slice2])
    
    for keys, df_slice in df_track2.groupby(['Manufacturer', 'Channel']):
        df_slice['training_till'] = training_till
        df_slice2 = pd.pivot(df_slice, index= ['Manufacturer', 'Channel', 'training_till'], columns= ['Date'],
                          values= ['units MAPE'])
        df_slice2.columns = [pd.to_datetime(m[1]).strftime("%B %Y") for m in df_slice2.columns]
        df_slice2 = df_slice2.reset_index()
        df_slice2['overall_wape'] = WAPE(df_slice['Units'], df_slice['units_pred'])
        df_res_units = pd.concat([df_res_units, df_slice2])

    return df_res_som, df_res_sales, df_res_units


def tdp_ppu_table(df_track4, df_res_ppu, df_res_tdp):
    """generates TDP and PPU error metrics table used in prediction accuracy page

    Args:
        df_track4 (dataframe)
        df_res_ppu (dataframe)
        df_res_tdp (dataframe)
    Returns:
        tuple of dataframes : df_res_ppu, df_res_tdp
    """  

    training_till = (df_track4['Date'].min()- relativedelta(months = 1)).strftime("%B %Y")
    for keys, df_slice in df_track4.groupby(['Manufacturer', 'Channel']):
        df_slice['training_till'] = training_till
        df_slice2 = pd.pivot(df_slice, index= ['Manufacturer', 'Channel', 'training_till'], columns= ['Date'],
                          values= ['PPU MAPE'])
        df_slice2.columns = [pd.to_datetime(m[1]).strftime("%B %Y") for m in df_slice2.columns]
        df_slice2 = df_slice2.reset_index()
        df_slice2['overall_wape'] = WAPE(df_slice['PPU'], df_slice['PPU_hat'])
        df_res_ppu = pd.concat([df_res_ppu, df_slice2])

    for keys, df_slice in df_track4.groupby(['Manufacturer', 'Channel']):
        df_slice['training_till'] = training_till
        df_slice2 = pd.pivot(df_slice, index= ['Manufacturer', 'Channel', 'training_till'], columns= ['Date'],
                          values= ['TDP MAPE'])
        df_slice2.columns = [pd.to_datetime(m[1]).strftime("%B %Y") for m in df_slice2.columns]
        df_slice2 = df_slice2.reset_index()
        df_slice2['overall_wape'] = WAPE(df_slice['TDP'], df_slice['TDP_hat'])
        df_res_tdp = pd.concat([df_res_tdp, df_slice2])

    return df_res_ppu, df_res_tdp


def kpi_prediction_best_model(data_new, kpi, predict_future_months, test_months=3, n_days=None):
    """Selects the best model for TDP/PPU prediction and predicts using the model

    Args:
        data_new (dataframe)
        kpi (str): TDP/PPU
        predict_future_months (int): no of months we want to predict for
        test_months (int, optional): no of months we want to use as test period in the existing dataframe to select best model. Defaults to 4.
        n_days (int, optional): used differently at different parts of code. Defaults to None.
    """    
    model_dict = {}
    
    data_new['Date'] = pd.to_datetime(data_new['Date'])
    data_new = data_new[['Manufacturer', 'Channel', 'Date', kpi]]

    data = data_new.copy()

    if n_days:
        data = data_new[~data_new['Date'].isin(sorted(data_new['Date'].unique())[-1*n_days:])]
        data_hold = data_new[data_new['Date'].isin(sorted(data_new['Date'].unique())[-1*n_days:])]
        
    df_pred = pd.DataFrame()
    for keys, df_grp in data.groupby(['Manufacturer', 'Channel']):
#         print(keys)
        mfg, cnl = keys
        df_grp = df_grp.sort_values(by='Date')
        df_grp = df_grp[df_grp[kpi]>0]
        df_grp = df_grp.reset_index(drop = True)
        mape_dict = {}
        wape_dict = {}

        ########################################################
        try:
            #exp smoothing
            train_data =  df_grp[~df_grp['Date'].isin(sorted(df_grp['Date'].unique())[-1*test_months:])]
            test_data  =  df_grp[df_grp['Date'].isin(sorted(df_grp['Date'].unique())[-1*test_months:])]
            train_data = train_data[['Date', kpi]].set_index('Date', drop=True)
            train_data[kpi] = train_data[kpi].astype('float64')                         
            train_data = train_data.to_period(freq="M")

            # modeling - exp smoothing
            model_1 = ExponentialSmoothing(trend='mul', seasonal='multiplicative', sp=12)
            model_1.fit(train_data)
            future_pred = pd.DataFrame(model_1.predict(fh=np.arange(1, test_months+1)))
            future_pred.columns = [f'{kpi}_hat_expsm']
            future_pred = future_pred.to_timestamp().reset_index().rename(columns= {'index': 'Date'})

            test_data = test_data.merge(future_pred, on= ['Date'], how= 'outer')
            test_mape_exp = MAPE(test_data[f'{kpi}'], test_data[f'{kpi}_hat_expsm'])
            test_wape_exp = WAPE(test_data[f'{kpi}'], test_data[f'{kpi}_hat_expsm'])

            #########################################################
            #Prophet
            df1 = df_grp[['Date', kpi]].rename(columns={kpi:'y', 'Date':'ds'})
            df1 = df1.sort_values(by='ds').reset_index(drop=True)
            train, test = df1.iloc[:-1*test_months], df1.iloc[-1*test_months:]

            # modeling
            model_2 = Prophet(daily_seasonality= False,
                    weekly_seasonality= False,
                    yearly_seasonality=False, # holidays = holiday,
                    ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
                            .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)
            model_2.fit(train)

            forecast_test = model_2.predict(test[['ds']]).merge(test[['ds', 'y']], on= ['ds'])

            forecast_test = forecast_test.reset_index().rename(columns = {'ds' : 'Date', 
                                                                        'y' : f'{kpi}', 
                                                                        'yhat' : f'{kpi}_hat_prophet'})

            test_data = test_data.merge(forecast_test[['Date', f'{kpi}_hat_prophet']], 
                                        on = ['Date'], how =  'outer')

            test_mape_pro = MAPE(test_data[f'{kpi}'], test_data[f'{kpi}_hat_prophet'])
            test_wape_pro = WAPE(test_data[f'{kpi}'], test_data[f'{kpi}_hat_prophet'])

            #########################################
            
            wape_dict['expsm_wape'] = test_wape_exp
            wape_dict['prophet_wape'] = test_wape_pro
        except:
            #Prophet
            df1 = df_grp[['Date', kpi]].rename(columns={kpi:'y', 'Date':'ds'})
            df1 = df1.sort_values(by='ds').reset_index(drop=True)
            train, test = df1.iloc[:-1*test_months], df1.iloc[-1*test_months:]

            # modeling
            model_2 = Prophet(daily_seasonality= False,
                    weekly_seasonality= False,
                    yearly_seasonality=False, # holidays = holiday,
                    ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
                            .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)
            model_2.fit(train)

            forecast_test = model_2.predict(test[['ds']]).merge(test[['ds', 'y']], on= ['ds'])

            forecast_test = forecast_test.reset_index().rename(columns = {'ds' : 'Date', 
                                                                        'y' : f'{kpi}', 
                                                                        'yhat' : f'{kpi}_hat_prophet'})

            test_data = test_data.merge(forecast_test[['Date', f'{kpi}_hat_prophet']], 
                                        on = ['Date'], how =  'outer')

            test_mape_pro = MAPE(test_data[f'{kpi}'], test_data[f'{kpi}_hat_prophet'])
            test_wape_pro = WAPE(test_data[f'{kpi}'], test_data[f'{kpi}_hat_prophet'])
            
            wape_dict['prophet_wape'] = test_wape_pro

        model_select = min(wape_dict, key = wape_dict.get)

        if model_select == 'expsm_wape':
            model = model_1
            train_data_new = df_grp[['Date', kpi]].set_index('Date', drop=True)
            
            train_data_new = train_data_new.to_period(freq="M")
            model.fit(train_data_new)

            future_pred = pd.DataFrame(model.predict(fh=np.arange(1, predict_future_months+1)))
            future_pred.columns = [f'{kpi}_hat']
            future_pred = future_pred.to_timestamp().reset_index().rename(columns= {'index': 'Date'})
            future_pred[['Manufacturer', 'Channel', 'Model']] = mfg, cnl, 'exponential_smoothing'
            df_grp = df_grp.merge(future_pred, on= ['Date', 'Manufacturer', 'Channel'], how= 'outer')

        elif model_select == 'prophet_wape':
            model = Prophet(daily_seasonality= False,
                weekly_seasonality= False,
                yearly_seasonality=False, #holidays = holiday,
                   ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
                         .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)

            train_data_new = df_grp[['Date', kpi]].rename(columns={kpi:'y', 'Date':'ds'})
            train_data_new = train_data_new.sort_values(by='ds').reset_index(drop=True)
            model.fit(train_data_new)
            future_data = model.make_future_dataframe( periods = predict_future_months, 
                                                      freq='m', include_history=False)
            future_data['ds'] = pd.DatetimeIndex(future_data['ds']) + pd.DateOffset(1)

            forecast = model.predict(future_data[['ds']])
            forecast = forecast[['ds', 'yhat']].rename(columns = {'ds': 'Date', 'yhat' : f'{kpi}_hat'})
            forecast[['Manufacturer', 'Channel', 'Model']] = mfg, cnl, 'prophet'
            df_grp = df_grp.merge(forecast, on = ['Date','Manufacturer', 'Channel' ], how = 'outer')
        
        ###############################
        if n_days:
            data_hold_comb = data_hold[(data_hold['Manufacturer'] == mfg) & (data_hold['Channel'] == cnl)]
            data_hold_comb = data_hold_comb.rename(columns = {kpi : f'{kpi}_act'})
            df_grp = df_grp.merge(data_hold_comb, on = ['Date','Manufacturer', 'Channel'], how = 'left')
            df_grp[kpi] = np.where(df_grp[kpi].isna(), df_grp[f'{kpi}_act'], df_grp[kpi])
            df_grp = df_grp.drop(columns= [f'{kpi}_act'])

        ################################

        df_pred = pd.concat([df_pred, df_grp])
        
    return df_pred


def all_subsets(ss):
    """generates all possible subsets of combinations from items in the list

    Args:
        ss (list)
    Returns:
        list of lists
    """ 
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

def final_processing(X, cols_to_select, coeff):
    """Calculate predictions by multiplying coefficients and actual values and adding intercept

    Args:
        X (dataframe)
        cols_to_select (list)
        coeff (dataframe)
    Returns:
        list (prediction) : new
    """ 
    X['units_pred'] = 0

    # import pdb;pdb.set_trace()
    for items in cols_to_select:
        if str(items).isnumeric():
            items = int(items)
        X['units_pred'] = X['units_pred'] + X[items] * coeff[coeff['features']==items]['coef'].values[0]
    X['units_pred'] = X['units_pred'] + coeff[coeff['features']=='intercept']['coef'].values[0]

    new = X['units_pred']

    return new

def constrained_sales_prediction(df_, sel_feat, lb, ub, target, forecast=False):
    """Fit the constrained regression model with bounds on the variables, 
    saves the coefficients from the model, 
    predicts future data

    Args:
        df_ (dataframe)
        sel_feat (list): list of selected features for model training
        lb (array): array of lower bounds on the variables
        ub (array): array of upper bounds on the variables
        target (string): target variable to predict
        forecast (bool, optional): whether to perform forecast on the future data or train on the training data

    Returns:
        tuple of dataframe: df_ --> sales prediction dataframe on 1 combination, 
        coeff --> model coefficients
    """   
    df_ = df_.sort_values(by= 'Date').reset_index(drop=True)
    assert df_['Channel'].nunique()==1
    cnl = df_['Channel'].unique()[0]
    df_2 = df_.copy()
    log_feat = [i for i in sel_feat]
    df_2[log_feat] = np.where(df_2[log_feat] > 0, df_2[log_feat].apply(np.log10), np.where(df_2[log_feat]<0,
                                                                                  df_2[log_feat],0))
    if forecast == False:
        df_temp = df_.copy()
        # Modelling
        df_1 = df_temp[sel_feat + [target]].copy()

        # log transforming variable
        df_1[log_feat] = np.where(df_1[log_feat] > 0, df_1[log_feat].apply(np.log10), np.where(df_1[log_feat]<0,
                                                                                  df_1[log_feat],0))
    
        # Convert independent variables to a matric
        X = df_1[sel_feat].iloc[:-1*test_months].values
        y = df_1[target].iloc[:-1*test_months].values
    else:
        
        df_temp = df_.sort_values(by='Date').iloc[:-1*future_months]
        # Modelling
        df_1 = df_temp[sel_feat + [target]].copy()

        # log transforming variable
        df_1[log_feat] = np.where(df_1[log_feat] > 0, df_1[log_feat].apply(np.log10), np.where(df_1[log_feat]<0,
                                                                                  df_1[log_feat],0))

        # Convert independent variables to a matric
        X = df_1[sel_feat].values
        y = df_1[target].values

    # Add an array of ones to act as bias coefficient
    ones = np.ones(X.shape[0])

    # Combine array of ones and indepedent variables
    X = np.concatenate(( ones[:,np.newaxis],X),axis=1)    

    results = lsq_linear(X,y,bounds=(lb,ub),lsmr_tol='auto')

    # Coefficients calculations
    new = []
    for i, items in enumerate(sel_feat):
        new.append(results['x'][i+1])

    coeff = pd.DataFrame({'features': list(sel_feat), 'coef': new}, dtype=object)
    coeff = pd.concat([coeff,
                    pd.DataFrame({'features': 'intercept', 'coef': results['x'][0]}, index=[0],
                                    dtype=object)], axis=0)
    coeff = coeff.reset_index(drop=True)

    y = final_processing(df_2, sel_feat, coeff)
    df_['units_pred'] = y
    mean_value = np.mean(y[-18:])
    df_['units_pred'] = np.where(df_['units_pred'] < 0, mean_value, df_['units_pred'])
    df_['sales_pred'] = df_['units_pred'] * df_['PPU']

    return df_, coeff

def sales_prediction_constrained_regression(data, future_months, test_months, target):
    """Perform training and forecasting using the constrained regression model with bounds on the variables on whole dataset, 
    saves the coefficients from the model, 
    predicts future data

    Args:
        data (dataframe)
        future_months (int): no of future months we want to predict
        test_months (int): no of months to be used as testing data in the existing data to select the best model
        target (string): target variable to predict

    Returns:
        tuple of dataframe: df_sales_pred --> sales prediction dataframe, 
        model_selected_feat --> modelname, channel and manufacturer and selected features are saved ,
        df_coef --> feature importance data is saved
        df_scores --> dataframe with trained model tracker for train test sets
    """  
    df_sales_pred = pd.DataFrame()
    model_selected_feat = pd.DataFrame(columns= ['combination', 'model', 'features'])
    df_coef = pd.DataFrame()
    df_scores = pd.DataFrame(columns = ['Month','Channel','Manufacturer','Model Selected', 
                                        'Training Period',f'Training Error ({target} WAPE in %)',
                                        "Testing Period",f'Testing Error ({target} WAPE in %)'])

    avg_price_lst, tdp_lst, avg_price_ratio_features, tdp_ratio_features = generate_regression_features(data)
    ratio_lst = avg_price_ratio_features + tdp_ratio_features

    for keys, df_slice in data.groupby(['Manufacturer', 'Channel']):
        mfg, cnl = keys
        print(keys)
        combination = cnl + '_' + mfg
        df_slice = df_slice.sort_values(by= 'Date').reset_index(drop=True)
        oh_cols = list(pd.get_dummies(df_slice['Month']).columns.values)
        encoded_features = pd.get_dummies(df_slice['Month'])
        df_slice2 = pd.concat([df_slice, encoded_features],axis=1)
        df_slice3 = df_slice2.sort_values(by='Date').iloc[:-1*future_months]
        min_date = df_slice3.Date.min()
        max_date = df_slice3.Date.max()
        train_period = min_date.strftime("%b %y") + " - " +(max_date - relativedelta(months=test_months)).strftime("%b %y")
        test_period = (max_date - relativedelta(months=test_months-1)).strftime("%b %y") + " - " + max_date.strftime("%b %y")
        
        # removing features not suitable for combination
        price_feat = [f'{mfg} PPU'] + [i for i in avg_price_lst if i not in [f'{mfg} PPU']]
        tdp_feat = [f'{mfg} TDP'] + [i for i in tdp_lst if i not in [f'{mfg} TDP']]

        # wape dict
        wape_dict = {}
        train_wape_dict = {}
        
        for comb in all_subsets(seasonality_cols):
            seasonality_feat = list(comb)
            sel_feat = price_feat + tdp_feat + seasonality_feat
            for intercept in params['intercept']:
                min_int, max_int = intercept
                lb = [min_int] + min_feat_bounds + seasonality_min_bounds
                lb = np.array(lb[:len(sel_feat)+1])
                ub = [max_int] + max_feat_bounds + seasonality_max_bounds
                ub = np.array(ub[:len(sel_feat)+1])
                df_pred, _ = constrained_sales_prediction(df_slice3, sel_feat, lb, ub, 'Units')
                wape_sales = WAPE(df_pred['Sales'].tail(test_months), df_pred['sales_pred'].tail(test_months))
                train_wape_sales = WAPE(df_pred['Sales'].iloc[:-test_months], df_pred['sales_pred'].iloc[:-test_months])
                train_wape_dict[intercept, comb] = train_wape_sales
                wape_dict[intercept, comb] = wape_sales
        model_select = min(wape_dict, key = wape_dict.get)
        intercept, comb = model_select
        min_int, max_int = intercept
        sel_feat = price_feat + tdp_feat + list(comb)
        lb = [min_int] + min_feat_bounds + seasonality_min_bounds
        lb = np.array(lb[:len(sel_feat)+1])
        ub = [max_int] + max_feat_bounds + seasonality_max_bounds
        ub = np.array(ub[:len(sel_feat)+1])
        df_pred, coeff = constrained_sales_prediction(df_slice2, sel_feat, lb, ub, 'Units', forecast=True)
        coeff[['Manufacturer', 'Channel']] = [mfg, cnl]
        df_coef = pd.concat([df_coef, coeff])
        df_sales_pred = pd.concat([df_sales_pred, df_pred])
        model_selected_feat.loc[len(model_selected_feat)] = [combination, "ConstrainedRegression", sel_feat]
        df_scores.loc[len(df_scores)] = [max_date.strftime("%b-%y"), cnl, mfg, "ConstrainedRegression", 
                                            train_period, train_wape_dict[model_select], test_period, wape_dict[model_select]]

        
    df_scores = df_scores.round(decimals = 2)
    df_sales_pred['total_sales_hat'] = df_sales_pred.groupby(['Date', 'Channel'])['sales_pred'].transform('sum')
    df_sales_pred['SOM_hat'] = df_sales_pred['sales_pred']*100/df_sales_pred['total_sales_hat']
    del df_sales_pred['total_sales_hat']
    df_sales_pred = df_sales_pred.drop(columns= oh_cols)
    return df_sales_pred, model_selected_feat, df_coef, df_scores

def get_cagr(df_cat,manufacturer,channel,col=None):   
    # Sort the DataFrame by year in ascending order
    df = df_cat[(df_cat['Channel']==channel) & (df_cat['Manufacturer']==manufacturer)].reset_index(drop=True)
    df = df.sort_values('Year')

    # Calculate the percentage change in sales from one year to the next
    df[f'pct_change_{col}'] = df[col].pct_change()

    # Calculate the compounded growth rate for each year
    df[f'compounded_growth_{col}'] = (1 + df[f'pct_change_{col}']).apply(np.log).cumsum().apply(np.exp)

    # Calculate the overall CAGR for the entire period
    cagr = (df[f'compounded_growth_{col}'].iloc[-1] / df[f'compounded_growth_{col}'].iloc[1]) ** (1 / (len(df) - 1)) - 1
    
    c = col.split("_")[-1]
    dev = df[f"{c}_dev"].max()
    
    return cagr, dev


def get_est(df_cat,manufacturer,channel, forecast_years=3, col=None,cagr=None):
    # Use the CAGR to forecast sales for the next 5 years
    # Sort the DataFrame by year in ascending order
    df = df_cat[(df_cat['Channel']==channel) & (df_cat['Manufacturer']==manufacturer)].reset_index(drop=True)
    df = df.sort_values('Year')
    starting_sales = df[f'{col}'].iloc[-1]

    # Calculate the estimated sales for each year in the forecast period
    forecast_sales = [starting_sales * (1 + cagr) ** n for n in range(1, forecast_years + 1)]

    # Create a new DataFrame to store the sales forecast
    forecast_df = pd.DataFrame({
        'Year': range(df['Year'].max() + 1, df['Year'].max() + forecast_years + 1),
        f'{col}': forecast_sales
    })

    # Print the forecasted sales for the next five years
#     print("Forecasted Sales for the Next Five Years:")
    return forecast_df


def calculate_sales(price, max_sales, max_price):
    if price > max_price:
        # Calculate the sales based on the linear decline function
        sales = max_sales * (1 - (price - max_price) / (price + max_price))
    else:
        pass
    return sales
