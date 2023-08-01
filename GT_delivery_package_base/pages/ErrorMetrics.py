import os
import colorlover
import pandas as pd
import numpy as np

from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dateutil.relativedelta import relativedelta

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import dash
import dash_table
from dash import dcc, html, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from utilities.constants import *
from utilities.utils import *


# Function for styling the table
def discrete_background_color_bins(df, n_bins=7):
    """this function color codes the performance matrix in Prediction accuracy page
    green (low error) --> red (high error)

    Args:
        df (dataframe)
        n_bins (int, optional): _description_. Defaults to 7.

    Returns:
        styles, html div: Color coded performance matrix
    """    
    bounds = [i * (1.0 / n_bins) for i in range(n_bins+1)]
    data = df.select_dtypes('number')
    df_max = data.max().max()
    df_min = data.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins+4)]['div']['RdYlGn'][2:-2][len(bounds) - i - 1]
        color = 'black'

        for column in data.columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
            styles.append({
                    'if': {
                        'filter_query': '{{{}}} is blank'.format(column),
                        'column_id': column
                    },
                    'backgroundColor': '#FFFFFF',
                    'color': 'white'
                })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))


# Importing error matrix data files 
df_sales = pd.read_csv(os.path.join(data_dir,sales_error_file))
df_units = pd.read_csv(os.path.join(data_dir,units_error_file))
df_som = pd.read_csv(os.path.join(data_dir,som_error_file))

df_ppu = pd.read_csv(os.path.join(data_dir,ppu_error_file))
df_tdp = pd.read_csv(os.path.join(data_dir,tdp_error_file))

# Rounding to 2 decimal places
df_sales = df_sales.round(decimals = 2)
df_units = df_units.round(decimals = 2)
df_som = df_som.round(decimals = 2)
df_ppu = df_ppu.round(decimals = 2)
df_tdp = df_tdp.round(decimals = 2)

# Importing predicted values for som, sales, ppu, tdp for error tracking
df_sales_req = pd.read_csv(os.path.join(data_dir,sales_req_file))
df_sales_req.rename(columns = {'units_pred' : 'Units_pred', 'sales_pred' : 'Sales_pred'}, inplace = True)

df_ppu_tdp_req = pd.read_csv(os.path.join(data_dir,ppu_tdp_req_file))
results = get_sales_kpi_values(df_sales_req)

results = get_kpi_values(df_ppu_tdp_req, results, col='PPU')
results = get_kpi_values(df_ppu_tdp_req, results, col='TDP')

# Select display columns in UI
display_sales_cols = df_sales.columns[2:]
display_sales_cols = [
        {'name': i, 'id': i, 'editable': False} for i in display_sales_cols
    ]

display_som_cols = df_som.columns[2:]
display_som_cols = [
        {'name': i, 'id': i, 'editable': False} for i in display_som_cols
    ]

#### Upper tile card values
df_sales_pred = pd.read_csv(os.path.join(data_dir, sales_pred_filename))
df_sales_pred['Date'] = pd.to_datetime(df_sales_pred['Date'])
df_sales_pred = df_sales_pred[df_sales_pred['Sales'].notna()]

max_date = df_sales_pred.Date.max()

#### Error plots
df_track = pd.DataFrame()
cols = ['Manufacturer', 'Channel', 'Date', 'SOM', 'SOM_hat', 'Sales', 'sales_pred']
for keys, df_slice in df_sales_pred.groupby(['Channel', 'Manufacturer']):
    df_slice = df_slice.sort_values(by='Date')[cols].tail(6)
    df_track = pd.concat([df_track, df_slice])

df_track2 = pd.DataFrame()
for keys, df_slice in df_track.groupby(['Manufacturer', 'Channel']):
    df_slice['sales MAPE'] = np.abs(df_slice['Sales'] - df_slice['sales_pred'])*100/np.abs(df_slice['Sales'])
    df_slice['SOM mae'] = np.abs(df_slice['SOM']-df_slice['SOM_hat'])
    df_track2 = pd.concat([df_track2, df_slice])
df_track2 = df_track2.sort_values(by=['Manufacturer', 'Channel', 'Date'])
df_track2 = df_track2[['Manufacturer', 'Channel', 'Date', 'SOM', 'SOM_hat', 'SOM mae',
                     'Sales', 'sales_pred', 'sales MAPE']]
df_track2['Date'] = df_track2['Date'].apply(lambda x: x.strftime("%B %Y"))


# Importing Feature Importance Data
df_feat_imp = pd.read_csv(os.path.join(data_dir, feature_importance_filename))

dash.register_page(__name__)

layout = dbc.Container([

    dbc.Row([
        dbc.ButtonGroup(
        [
            dbc.Button("Home", id= 'navigation-page',
                       style = {'width': '3in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/home'),
            dbc.Button("Baseline Prediction", id= 'first-page',
                       style = {'width': '3in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/Simulation',
                       className="ml-auto"),
            dbc.Button("Prediction Accuracy", className="ml-auto", id= 'second-page',
                       style = {'width': '3in', 'background-color': '#01529C', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'}
                       ),
            dbc.Button("Scenario Management ", className="ml-auto", id= 'third-page',
                       style = {'width': '3in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/scenario')
        ],
    )
    ], align = 'center', justify= 'center'),

    dbc.Row([
       dbc.Col([
           html.P(f"Data is available till {max_date.strftime('%b')}-{max_date.strftime('%Y')}",
                  style= {'font-family': 'Verdana', 'font-size': '13px'}),
       ], width=3)
    ], justify="right", align="right"),

    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Label('Channel', style= {'marginLeft': '0px', 'marginRight': '64px', 'font-family': 'Verdana', 'font-size': '14px',
                                          'font-weight': 'bold'}),
            dcc.Dropdown(id='chosen-channel',
                options=[{"label": i, "value": i} for i in df_sales.Channel.unique()],
                         value='OT', clearable = False, style={'color': 'black', 'font-family': 'Verdana', 'font-size': '14px'}),
        ], width=2),

        dbc.Col([
            html.Label('Manufacturer', style= {'marginLeft': '0px', 'marginRight': '50px', 'font-family': 'Verdana', 'font-size': '14px',
                                          'font-weight': 'bold'}),
            dcc.Dropdown(id='chosen-manufacturer',
                options=[{"label": i, "value": i} for i in df_sales.Manufacturer.unique()],
                         value='PEPSICO', clearable= False, style={'color': 'black', 'font-family': 'Verdana', 'font-size': '14px'}),
        ], width=2),
        dbc.Col([
            html.Label('Parameter', style= {'marginLeft': '0px', 'marginRight': '50px', 'font-family': 'Verdana', 'font-size': '14px',
                                          'font-weight': 'bold'}),
            dcc.Dropdown(id='chosen-parameter',
                        options=[{"label": i, "value": i} for i in ['SOM', 'SALES', 'UNITS']],
                                 value='SOM', clearable= False, style={'color': 'black',
                                                                      'marginLeft': '0px',
                                                                          'width': '210px', 'font-family': 'Verdana', 'font-size': '14px'})
        ])

    ]),
    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ], style= {'border-bottom': '2px solid grey', 'margin': '0px'}),

    dbc.Row([
        dbc.Col([

            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children= f"Model Performance - Evaluation Till {max_date.strftime('%b-%Y')}", style = {'font-weight': 'bold',
                                                                                       'color': '#1D4693', 'font-size': '100%',
                                                                                       'border': 'none', 'background-color': '#E8E8E8',
                                                                                      'width': '800px', 'font-family': 'Verdana'})
                ], horizontal=True),

                dbc.ListGroup([
                    dbc.ListGroupItem(children= '', style = {'width': '180px', 'font-weight': 'bold', 'color': '#F9F9F9',
                                                                    'font-color': 'white', 'border': 'none',
                                                                    'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '13px'}),
                    dbc.ListGroupItem(children= 'Current Month', style = {'font-weight': 'bold', 'border': 'none', 'text-align': 'center',
                                                                         'width': '160px', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '13px'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(children= 'Last 6 Months', style = {'font-weight': 'bold', 'width': '160px', 'text-align': 'right',
                                                                          'border': 'none', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '13px'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(children= 'Last 12 Months', style = {'font-weight': 'bold', 'border': 'none', 'text-align': 'center', 'width': '170px',
                                                                           'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '13px'}),
                ], horizontal=True),

                dbc.ListGroup([
                    dbc.ListGroupItem(children= '', style = {'width': '180px', 'font-weight': 'bold', 'color': '#F9F9F9',
                                                                    'font-color': 'white', 'border': 'none',
                                                                    'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '13px'}),
                    dbc.ListGroupItem(id= 'last-month-title', style = {'font-weight': 'bold', 'border': 'none', 'text-align': 'center',
                                                                         'width': '160px', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '13px'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= 'last-6-months-title', style = {'font-weight': 'bold', 'width': '160px', 'text-align': 'center',
                                                                          'border': 'none', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '13px'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= 'last-12-months-title', style = {'font-weight': 'bold', 'border': 'none', 'text-align': 'center', 'width': '160px',
                                                                           'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '13px'}),
                ], horizontal=True),

                dbc.ListGroup([
                    dbc.ListGroupItem(id= 'param-text', style= {'width': '180px', 'border': 'none', 'font-weight': 'bold',
                                                                'background-color': '#F9F9F9', 'font-family': 'Verdana', 'font-size': '13px'}),
                    dbc.ListGroupItem(id= 'last-month', style = {'background-color': '#01529C', 'width': '160px',
                                                                 'font-weight': 'bold', 'color': 'white',
                                                                    'font-size': '13px', 'border-radius': '5px',
                                                                    'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'},
                                      ),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= '6-months', style = {'background-color': '#01529C', 'width': '160px',
                                                               'color': 'white',
                                                               'font-weight': 'bold',
                                                               'font-size': '13px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(id= '12-months', style = {'background-color': '#01529C', 'width': '160px',
                                                               'color': 'white',
                                                               'font-weight': 'bold', 'width': '160px',
                                                               'font-size': '13px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'}),

                ], horizontal=True),
                html.Br(),
                dbc.ListGroup([
                    dbc.ListGroupItem(children= 'PPU Error (%)', style= {'width': '180px', 'border': 'none', 'font-weight': 'bold',
                                                                'background-color': '#F9F9F9', 'font-family': 'Verdana', 'font-size': '13px'}),
                    dbc.ListGroupItem(id= 'last-month-ppu', style = {'background-color': '#01529C', 'width': '160px',
                                                                 'font-weight': 'bold', 'color': 'white',
                                                                    'font-size': '13px', 'border-radius': '5px',
                                                                    'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'},
                                      ),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= '6-months-ppu', style = {'background-color': '#01529C', 'width': '160px',
                                                               'color': 'white',
                                                               'font-weight': 'bold',
                                                               'font-size': '13px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(id= '12-months-ppu', style = {'background-color': '#01529C', 'width': '160px',
                                                               'color': 'white',
                                                               'font-weight': 'bold', 'width': '160px',
                                                               'font-size': '13px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'}),

                ], horizontal=True),
                html.Br(),
                dbc.ListGroup([
                    dbc.ListGroupItem(children= 'TDP Error (%)', style= {'width': '180px', 'border': 'none', 'font-weight': 'bold',
                                                                'background-color': '#F9F9F9', 'font-family': 'Verdana', 'font-size': '13px'}),
                    dbc.ListGroupItem(id= 'last-month-tdp', style = {'background-color': '#01529C', 'width': '160px',
                                                                 'font-weight': 'bold', 'color': 'white',
                                                                    'font-size': '13px', 'border-radius': '5px',
                                                                    'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'},
                                      ),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= '6-months-tdp', style = {'background-color': '#01529C', 'width': '160px',
                                                               'color': 'white',
                                                               'font-weight': 'bold',
                                                               'font-size': '13px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(id= '12-months-tdp', style = {'background-color': '#01529C', 'width': '160px',
                                                               'color': 'white',
                                                               'font-weight': 'bold', 'width': '160px',
                                                               'font-size': '13px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'}),

                ], horizontal=True),


            ], style= {'border': 'none', 'background-color': '#F9F9F9'}

            ),
        ], width= 6),
        dbc.Col([
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children= 'Model Performance Trend', style = {'font-weight': 'bold',
                                                                                       'color': '#1D4693', 'font-size': '100%',
                                                                                       'border': 'none', 'background-color': '#E8E8E8',
                                                                                      'width': '800px', 'font-family': 'Verdana',
                                                                                      'margin-top': '0px'})
                ], horizontal=True),
                dbc.ListGroupItem(id= 'error-graph-text', style= {'border': 'none', 'font-weight': 'bold',
                                                                'background-color': '#F9F9F9', 'font-family': 'Verdana', 'font-size': '13px'}),
                dcc.Graph(id='error-graph', style= {'height': '40vh'})
            ], style= {'height': '40vh', 'marginLeft': '10px', 'border': 'none',
                       'background-color': '#F9F9F9', 'margin-top': '0px'})
        ], width= 6),
    ]),

    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ], style= {'border-bottom': '2px solid grey', 'margin': '0px'}),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children= 'Model Performance Matrix', style= {'width': '100%', 'color': '#1D4693',
                                                                                      'background-color': '#E8E8E8', 'border': 'none',
                                                                                     'font-weight': 'bold', 'font-size': '100%', 'font-family': 'Verdana'}),
                ])

            ], style= {'border': 'none'}

            ),
        ], width= 12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dash_table.DataTable(id='error-table',
                                style_header={ 'border': '1px solid white', 'whiteSpace':'normal', 'color': 'white',
                                              'font-weight': 'bold', 'backgroundColor': '#01529C', 'font-family': 'Verdana',
                                              'font-size':'10px'},
                                style_cell={ 'minWidth': 80, 'maxWidth': 120, 'border': '1px solid grey',
                                            'font-family': 'Verdana', 'font-size':'10px'},
                                style_table={'overflowX': 'auto', 'height': '300px'},
                                fixed_rows={'headers': True},
                                )
        ], width=12, style= {'font-size': '11px'}, align="center")
    ]),


], fluid = True, style = {'background-color': '#F9F9F9'})

# Based on selected filters (channel and manufacturer) --> Last Month Sep-2022 Model Performance values are updated
@callback(
    Output("param-text", 'children'),
    Output("last-month", 'children'),
    Output("6-months", 'children'),
    Output("12-months", 'children'),
    Output("last-month-title", 'children'),
    Output("last-6-months-title", 'children'),
    Output("last-12-months-title", 'children'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value'),
    Input('chosen-parameter', 'value')
)
def card_values(mfg, cnl, parameter):
    global df_sales_pred, df_sales,df_units, df_som, results
    """This function shows sales WAPE and som MAE values to be used in cards indicating error 
    values of last month, last 6 months and last 12 months combined in prediction accuracy page.

    Returns:
        tuple : text, last_month, rolling_6_months, rolling_12_months, last_month_title, last_6_month_title, last_12_month_title
        last_month: error in last_month's sales/SOM prediction considering prediction generated from data removing the last month's data
        rolling_6_months: Combined error in sales/SOM of 6 next immediate month prediction after removing last 1,2,..,6 months from the dataframe
        rolling_12_months: Combined error in sales/SOM of 12 next immediate month prediction after removing last 1,2,..,12 months from the dataframe
        last_month_title :  Text indicating the last_month name.
        last_6_month_title : Text indicating the rolling_6_months window used name.
        last_12_month_title :  Text indicating the rolling_12_months window used name.
    """
    
    combination = mfg + "_" + cnl

    df_temp = df_sales_pred[(df_sales_pred['Manufacturer']==mfg)&(df_sales_pred['Channel']==cnl)]

    df_error_sales = df_sales[(df_sales['Manufacturer']==mfg)&(df_sales['Channel']==cnl)]
    df_error_sales['date'] = pd.to_datetime(df_error_sales['Training Upto'], format= "%B %Y")
    df_error_sales = df_error_sales.sort_values(by= 'date', ascending=True).tail(1)
    last_month_title = (pd.to_datetime(df_error_sales['date'].values[0])+relativedelta(months=1)).strftime("%b %y")
    last_6_month_title = (pd.to_datetime(df_error_sales['date'].values[0])-relativedelta(months=4)).strftime("%b %y") + '-' + last_month_title
    last_12_month_title = (pd.to_datetime(df_error_sales['date'].values[0])-relativedelta(months=10)).strftime("%b %y")+ '-' + last_month_title
    
    df_error_units = df_units[(df_units['Manufacturer']==mfg)&(df_units['Channel']==cnl)]
    df_error_units['date'] = pd.to_datetime(df_error_units['Training Upto'], format= "%B %Y")
    df_error_units = df_error_units.sort_values(by= 'date', ascending=True).tail(1)

    df_error_som = df_som[(df_som['Manufacturer']==mfg)&(df_som['Channel']==cnl)]
    df_error_som['date'] = pd.to_datetime(df_error_som['Training Upto'], format= "%B %Y")
    df_error_som = df_error_som.sort_values(by= 'date', ascending=True).tail(1)

    if parameter == 'SOM':
        text = 'SOM MAE (%)'
        # last month
        last_month = df_error_som.iloc[:, -3].values[0]
        last_month = '{:.2f}'.format(last_month)

        # last 6 months
        rolling_6_months = results[combination][2]
        rolling_6_months = '{:.2f}'.format(rolling_6_months)

        # last 12 months
        rolling_12_months = results[combination][5]
        rolling_12_months = '{:.2f}'.format(rolling_12_months)
    elif parameter == 'SALES':
        text = 'Sales WAPE (%)'
        # last month
        last_month = df_error_sales.iloc[:, -3].values[0]
        last_month = '{:.2f}'.format(last_month)
        
        # last 6 months
        rolling_6_months = results[combination][1]
        rolling_6_months = '{:.2f}'.format(rolling_6_months)

        # last 12 months
        rolling_12_months = results[combination][4]
        rolling_12_months = '{:.2f}'.format(rolling_12_months)

    else:
        text = 'Units WAPE (%)'
        # last month
        last_month = df_error_units.iloc[:, -3].values[0]
        last_month = '{:.2f}'.format(last_month)

        # last 6 months
        rolling_6_months = results[combination][0]
        rolling_6_months = '{:.2f}'.format(rolling_6_months)

        # last 12 months
        rolling_12_months = results[combination][3]
        rolling_12_months = '{:.2f}'.format(rolling_12_months)

    return text, last_month, rolling_6_months, rolling_12_months, last_month_title, last_6_month_title, last_12_month_title


# Based on selected filters (channel and manufacturer) --> Last Month Sep-2022 Model Performance values are updated
@callback(
    Output('last-month-ppu', 'children'),
    Output('6-months-ppu', 'children'),
    Output('12-months-ppu', 'children'),
    Output('last-month-tdp', 'children'),
    Output('6-months-tdp', 'children'),
    Output('12-months-tdp', 'children'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value')
)
def ppu_tdp_card_values(mfg, cnl):
    """This function generates PPU and TDP WAPE values to be used in cards indicating error 
    values in last month, last 6 months and last 12 months combined in prediction accuracy page.

    Returns:
        tuple: last_month_ppu, last_6_months_ppu, last_12_months_ppu, last_month_tdp, last_6_months_tdp, last_12_months_tdp
        last_month_ppu: error in last_month PPU prediction considering prediction generated from data removing the last month's data
        last_6_months_ppu: Combined error of 6 next immediate month PPU prediction after removing last 1,2,..,6 months from the dataframe
        last_12_months_ppu: Combined error of 12 next immediate month PPU prediction after removing last 1,2,..,12 months from the dataframe
        last_month_tdp :  error in last_month TDP prediction considering prediction generated from data removing the last month's data.
        last_6_months_tdp : Combined error of 6 next immediate month TDP prediction after removing last 1,2,..,6 months from the dataframe
        last_12_months_tdp : Combined error of 12 next immediate month TDP prediction after removing last 1,2,..,12 months from the dataframe
    """
    global df_ppu, df_tdp, results

    combination = mfg + "_" + cnl

    df_error_ppu = df_ppu[(df_ppu['Manufacturer']==mfg)&(df_ppu['Channel']==cnl)]
    df_error_ppu['date'] = pd.to_datetime(df_error_ppu['Training Upto'], format= "%B %Y")
    df_error_ppu = df_error_ppu.sort_values(by= 'date', ascending=True).tail(1)

    df_error_tdp = df_tdp[(df_tdp['Manufacturer']==mfg)&(df_tdp['Channel']==cnl)]
    df_error_tdp['date'] = pd.to_datetime(df_error_tdp['Training Upto'], format= "%B %Y")
    df_error_tdp = df_error_tdp.sort_values(by= 'date', ascending=True).tail(1)

    last_month_ppu = df_error_ppu.iloc[:, -3].values[0]
    last_month_ppu = '{:.2f}'.format(last_month_ppu)

    last_month_tdp = df_error_tdp.iloc[:, -3].values[0]
    last_month_tdp = '{:.2f}'.format(last_month_tdp)

    # last 6 months ppu
    last_6_months_ppu = results[combination][6]
    last_6_months_ppu = '{:.2f}'.format(last_6_months_ppu)

    # last 12 months ppu
    last_12_months_ppu = results[combination][7]
    last_12_months_ppu = '{:.2f}'.format(last_12_months_ppu)

    # last 6 months tdp
    last_6_months_tdp = results[combination][8]
    last_6_months_tdp = '{:.2f}'.format(last_6_months_tdp)

    # last 12 months tdp
    last_12_months_tdp = results[combination][9]
    last_12_months_tdp = '{:.2f}'.format(last_12_months_tdp)

    return last_month_ppu, last_6_months_ppu, last_12_months_ppu, last_month_tdp, last_6_months_tdp, last_12_months_tdp


# Based on selected filters (Channel and Manufacturer) --> Last 6 Months Model Performance Trend is plotted
@callback(
    Output('error-graph-text', 'children'),
    Output('error-graph', 'figure'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value'),
    Input('chosen-parameter', 'value')
)
def card_plot(mfg, cnl, parameter):
    """This function takes ppu, tdp, sales and som error tracker files generated from generate error metrics and then 
    plots Last 6 Months Model Performance Trend
    Args:
        mfg (str):  Manufacturer
        cnl (str):  Channel
        parameter (str): SOM/Sales

    Returns:
        fig_text: Text written over PPU , TDP and sales/SOM plot 
        fig: Line chart representing Last 6 Months Model Performance Trend
    """    
    global  df_feat_imp, df_ppu, df_tdp, df_sales, df_units, df_som
    temp_sales = df_sales[(df_sales['Manufacturer']==mfg)&(df_sales['Channel']==cnl)]
    temp_sales = pd.DataFrame(np.diag(temp_sales.iloc[:, 3:]),
             index=[temp_sales.iloc[:, 3:-1].columns]).reset_index()
    temp_sales['date'] = pd.to_datetime(temp_sales['level_0'], format= "%B %Y")
    temp_sales.columns = ['Date', 'sales_error', 'date_format']
    temp_sales = temp_sales.tail(6)
    
    temp_units = df_units[(df_units['Manufacturer']==mfg)&(df_units['Channel']==cnl)]
    temp_units = pd.DataFrame(np.diag(temp_units.iloc[:, 3:]),
             index=[temp_units.iloc[:, 3:-1].columns]).reset_index()
    temp_units['date'] = pd.to_datetime(temp_units['level_0'], format= "%B %Y")
    temp_units.columns = ['Date', 'units_error', 'date_format']
    temp_units = temp_units.tail(6)

    temp_som = df_som[(df_som['Manufacturer']==mfg)&(df_som['Channel']==cnl)]
    temp_som = pd.DataFrame(np.diag(temp_som.iloc[:, 3:]),
             index=[temp_som.iloc[:, 3:-1].columns]).reset_index()
    temp_som['date'] = pd.to_datetime(temp_som['level_0'], format= "%B %Y")
    temp_som.columns = ['Date', 'som_error', 'date_format']
    temp_som = temp_som.tail(6)

    temp_ppu = df_ppu[(df_ppu['Manufacturer']==mfg)&(df_ppu['Channel']==cnl)]
    temp_ppu = pd.DataFrame(np.diag(temp_ppu.iloc[:, 3:]),
             index=[temp_ppu.iloc[:, 3:-1].columns]).reset_index()
    temp_ppu['date'] = pd.to_datetime(temp_ppu['level_0'], format= "%B %Y")
    temp_ppu.columns = ['Date', 'ppu_error', 'date_format']
    temp_ppu = temp_ppu.tail(6)

    temp_tdp = df_tdp[(df_tdp['Manufacturer']==mfg)&(df_tdp['Channel']==cnl)]
    temp_tdp = pd.DataFrame(np.diag(temp_tdp.iloc[:, 3:]),
             index=[temp_tdp.iloc[:, 3:-1].columns]).reset_index()
    temp_tdp['date'] = pd.to_datetime(temp_tdp['level_0'], format= "%B %Y")
    temp_tdp.columns = ['Date', 'tdp_error', 'date_format']
    temp_tdp = temp_tdp.tail(6)

    temp_sales = temp_sales.merge(temp_ppu, on= ['Date'], how= 'left').merge(temp_tdp, on= ['Date'], how= 'left')
    temp_units = temp_units.merge(temp_ppu, on= ['Date'], how= 'left').merge(temp_tdp, on= ['Date'], how= 'left')
    temp_som = temp_som.merge(temp_ppu, on= ['Date'], how= 'left').merge(temp_tdp, on= ['Date'], how= 'left')
    # print(temp_sales.head())
    if parameter == 'SOM':
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x= temp_som['Date'], y = temp_som['som_error'],
                                     name= 'SOM Errors (%)', marker= dict(color= '#01529C')), secondary_y=False)
        fig.add_trace(go.Scatter(x= temp_som['Date'], y = temp_som['ppu_error'],
                             name= 'PPU Errors (%)', marker= dict(color= '#00984A')),
             secondary_y=True)
        fig.add_trace(go.Scatter(x= temp_som['Date'], y = temp_som['tdp_error'],
                             name= 'TDP Errors (%)', marker= dict(color= '#EB7B30')),
             secondary_y=True)
        fig.update_layout(yaxis_title = 'SOM MAE (%)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                              margin= {'t': 0, 'b': 0, 'r': 0, 'l': 0}, legend=dict(orientation="h"))
        fig.update_yaxes(title_text="PPU/TDP Errors (%)", secondary_y=True)
        fig_text = f'{mfg} X {cnl} SOM Error (MAE %)'
    elif parameter == 'SALES':
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x= temp_sales['Date'], y = temp_sales['sales_error'],
                                     name= 'Sales Errors (%)', marker= dict(color= '#01529C')))
        fig.add_trace(go.Scatter(x= temp_sales['Date'], y = temp_sales['ppu_error'],
                             name= 'PPU Errors (%)', marker= dict(color= '#00984A')),
             secondary_y=True)
        fig.add_trace(go.Scatter(x= temp_sales['Date'], y = temp_sales['tdp_error'],
                             name= 'TDP Errors (%)', marker= dict(color= '#EB7B30')),
             secondary_y=True)
        fig.update_layout(yaxis_title = 'Sales WAPE (%)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                                margin= {'t': 0, 'b': 0, 'r': 0, 'l': 0}, legend=dict(orientation="h"))
        fig.update_yaxes(title_text="PPU/TDP Errors (%)", secondary_y=True)

        fig_text = f'{mfg} X {cnl} Sales Error (WAPE %)'
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x= temp_units['Date'], y = temp_units['units_error'],
                                     name= 'Units Errors (%)', marker= dict(color= '#01529C')))
        fig.add_trace(go.Scatter(x= temp_units['Date'], y = temp_units['ppu_error'],
                             name= 'PPU Errors (%)', marker= dict(color= '#00984A')),
             secondary_y=True)
        fig.add_trace(go.Scatter(x= temp_units['Date'], y = temp_units['tdp_error'],
                             name= 'TDP Errors (%)', marker= dict(color= '#EB7B30')),
             secondary_y=True)
        fig.update_layout(yaxis_title = 'Units WAPE (%)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                                margin= {'t': 0, 'b': 0, 'r': 0, 'l': 0}, legend=dict(orientation="h"))
        fig.update_yaxes(title_text="PPU/TDP Errors (%)", secondary_y=True)

        fig_text = f'{mfg} X {cnl} Units Error (WAPE %)'

    return fig_text, fig

# Based on selected filters (Channel and Manufacturer) --> Model Performance Matrix
@callback(
    Output('error-table', 'data'),
    Output('error-table', 'columns'),
    Output('error-table', 'style_data_conditional'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value'),
    Input('chosen-parameter', 'value')
)
def actualize_db(mfg, cnl, param):
    """based on the selected filters (channel, manufacturer and SOM/Sales)
    this function shows the Model Performance Matrix

    Args:
        mfg (str): selected from filters
        cnl (str): selected from filters
        param (str): selected from filters

    Returns:
        tuple: df_slice.to_dict('records') --> dataframe , (display_cols, styles --> styling)
    """    
    global display_sales_cols, display_som_cols, df_sales, df_units, df_som
    if param == 'SOM':
        df_slice = df_som[(df_som['Manufacturer']==mfg)&(df_som['Channel']==cnl)]
        display_cols = display_som_cols
        (styles, legend) = discrete_background_color_bins(df_slice)
    elif param == 'SALES':
        df_slice = df_sales[(df_sales['Manufacturer']==mfg)&(df_sales['Channel']==cnl)]
        display_cols = display_sales_cols
        (styles, legend) = discrete_background_color_bins(df_slice)
        
    else:
        df_slice = df_units[(df_units['Manufacturer']==mfg)&(df_units['Channel']==cnl)]
        display_cols = display_sales_cols
        (styles, legend) = discrete_background_color_bins(df_slice)

    return df_slice.to_dict('records'), display_cols, styles
