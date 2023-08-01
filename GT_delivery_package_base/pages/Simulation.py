# Importing required libraries
import os
import datetime
import pandas as pd
import numpy as np
import joblib
import math
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
import plotly.express as px
from utilities.constants import *
from utilities.utils import *

## Importing libraries useful for creating dash app
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash_table
from dateutil.relativedelta import relativedelta
from dash import callback

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)

# Importing Scantrack Base Data
df = pd.read_csv(os.path.join(data_dir, base_file))
del df['date']
df = df[df['channel']!='TOTAL']
df = df.rename(columns= {'date_new': 'date'})
df['date'] = pd.to_datetime(df['date'])
df.columns = ['Manufacturer', 'PPU', 'TDP', 'Units', 'Sales', 'Date', 'Channel', 'SOM']
max_date = df['Date'].max()
print(df.shape)

# Importing realistic simulation data
realistic_df = pd.read_csv(os.path.join(data_dir, realistic_simulation_file))

# Extracting generated feature names
avg_price_features, tdp_features, avg_price_ratio_features, tdp_ratio_features = generate_regression_features(df)
regression_features = avg_price_features + tdp_features + avg_price_ratio_features + tdp_ratio_features


# Importing Regressors Data having future predictions
df_reg_pred = pd.read_csv(os.path.join(data_dir, ppu_tdp_pred_filename))
df_reg_pred = df_reg_pred[['Manufacturer', 'Channel', 'Date', 'PPU', 'TDP']]
df = pd.concat([df, df_reg_pred])
print(df.shape)

# Performing feature generation
df_comp = generate_comp_features_1(df)
df_comp = df_comp.round(decimals = 2)
df_comp['Date'] = pd.to_datetime(df_comp['Date'])
df_comp['Month'] = df_comp['Date'].dt.month
###########################################################################################

# Importing Forecasted Sales data having next 18 months predictions
df_sales_pred = pd.read_csv(os.path.join(data_dir, sales_pred_filename))
df_sales_pred['Date'] = pd.to_datetime(df_sales_pred['Date'])
df_sales_pred = df_sales_pred.round(decimals = 2)

# Merging with External datasets
df_comp = df_comp.merge(df_sales_pred[['Date', 'Manufacturer', 'Channel']+ seasonality_cols], 
                        on= ['Date', 'Manufacturer', 'Channel'], how= 'left')
print(df_comp.shape)
df_comp = round(df_comp,2)

# Aggregating sales predicted/actual data across Manufacturers to get data for overall channels
df_all = df_sales_pred.groupby(['Manufacturer', 'Date'])[['Units', 'Sales', 'sales_pred', 'units_pred']].sum().reset_index()
df_all['Channel'] = 'TOTAL'
df_all['total_sales'] = df_all.groupby(['Date', 'Channel'])['Sales'].transform('sum')
df_all['SOM'] = df_all['Sales']*100/df_all['total_sales']
del df_all['total_sales']
df_all['total_sales_hat'] = df_all.groupby(['Date', 'Channel'])['sales_pred'].transform('sum')
df_all['SOM_hat'] = df_all['sales_pred']*100/df_all['total_sales_hat']
del df_all['total_sales_hat']
df_all['Month'] = df_all['Date'].dt.month

# Generating template data dynamically to be used in scenario management page
df_sales_pred['Date'] = pd.to_datetime(df_sales_pred['Date'])
df_next = df_sales_pred[df_sales_pred['Sales'].isnull()]
avg_cols_, tdp_cols_, _, _ = generate_regression_features(df_next)
comp_cols_ = avg_cols_ + tdp_cols_
df_next = df_next[['Manufacturer', 'Channel', 'Date']+comp_cols_]
inv_month_dict = {v: k for k, v in month_dict.items()}
df_next['key'] = df_next['Date'].dt.strftime("%B")
df_next['key'] = df_next['key'].str.upper()
df_next['key'] = df_next['key'].map(inv_month_dict)
df_next['year'] = df_next['Date'].dt.year
df_next['key'] = df_next['key'].astype(str) + " " + df_next['year'].astype(str)

xl_writer = pd.ExcelWriter(os.path.join(data_dir, template_filename),engine='xlsxwriter')
for channel in df_next.Channel.unique():
    save_double_column_df(get_channel_df(df_next, channel), xl_writer, sheet_name = channel)
xl_writer.close()


# Importing Feature Importance and model selected features datasets
df_coef = pd.read_csv(os.path.join(data_dir, feature_importance_filename))
df_coef['features'] = df_coef['features'].apply(lambda x: int(x) if str(x).isnumeric() else x)
df_feat_imp = feature_importance(df_coef)

model_features_data = pd.read_csv(os.path.join(data_dir,selected_features_filename))
model_features_data = process_features_file(model_features_data)

# Separating sets of columns to be shown in dash datatable having no edit access
display_cols = [
         {'name': 'Manufacturer', 'id': 'Manufacturer', 'editable': False},
         {'name': 'Channel', 'id': 'Channel', 'editable': False},
         {'name': 'Date', 'id': 'Date', 'editable': False},
         {'name': 'Month', 'id': 'Month', 'editable': False},
         {'name': 'Units', 'id': 'Units', 'editable': False},
         {'name': 'Sales', 'id': 'Sales', 'editable': False},
         {'name': 'SOM', 'id': 'SOM', 'editable': False}
            ]

# Registering Simulation as main page in the app
dash.register_page(__name__, path='/')

# page contents for the dash app
layout = dbc.Container([

    dbc.Row([
        dbc.ButtonGroup(
        [
            dbc.Button("Home", id= 'navigation-page',
                       style = {'width': '3in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/home'),
            dbc.Button("Baseline Prediction", id= 'first-page',
                       style = {'width': '3in', 'background-color': '#01529C', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       className="ml-auto"),
            dbc.Button("Prediction Accuracy", className="ml-auto", id= 'second-page',
                       style = {'width': '3in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/ErrorMetrics'),
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

#############################################################

    dbc.Row([
        dbc.Col([
            html.Label('Channel', style= {'marginLeft': '0px', 'marginRight': '64px', 'font-family': 'Verdana', 'font-size': '14px',
                                          'font-weight': 'bold'}),
            dcc.Dropdown(id='chosen-channel',
                options=[{"label": i, "value": i} for i in df_comp.Channel.unique().tolist() + ['TOTAL']],
                         value='OT', clearable = False, style={'color': 'black', 'font-family': 'Verdana', 'font-size': '14px'}),
            html.Br(),
            html.Img(id= 'channel-logo', width="180", height="150", 
                     style= {'margin-left': '2vh', 'text-align': 'center'})
        ], width= 2),
        dbc.Col([
            html.Label('Manufacturer', style= {'marginLeft': '0px', 'marginRight': '50px', 'font-family': 'Verdana', 'font-size': '14px',
                                                'font-weight': 'bold'}),
            dcc.Dropdown(id='chosen-manufacturer',
                options=[{"label": i, "value": i} for i in df_comp.Manufacturer.unique()],
                         value='PEPSICO', clearable= False, style={'color': 'black', 'font-family': 'Verdana', 'font-size': '14px'}),
            html.Br(),
            html.Img(id= 'manufacturer-logo', width="180", height="150",
                     style= {'margin-left': '2vh', 'text-align': 'center'})
        ], width = 2),
        dbc.Col([
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children= '', style = {'width': '40px', 'font-weight': 'bold', 'color': '#F9F9F9',
                                                                    'font-color': 'white', 'border': 'none',
                                                                    'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '13px'}),
                    dbc.ListGroupItem(id= 'som-ytd', style = {'font-weight': 'bold', 'border': 'none', 'text-align': 'right',
                                                                         'width': '130px', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '14px'}),
                    dbc.ListGroupItem(id = 'som-ytd-change'),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '1px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= 'som-next', style = {'font-weight': 'bold', 'width': '130px', 'text-align': 'right',
                                                                          'border': 'none', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '14px'}),
                    dbc.ListGroupItem(id= 'som-next-change'),
                    dbc.ListGroupItem(style = {'border': 'none','width': '1px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= 'som-next-year', style = {'font-weight': 'bold', 'border': 'none', 'text-align': 'right', 'width': '130px',
                                                                           'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '14px'}),
                    dbc.ListGroupItem(id= 'som-next-year-change'),
                ], horizontal=True),

                dbc.ListGroup([
                    dbc.ListGroupItem(children= '', style= {'width': '40px', 'border': 'none', 'font-weight': 'bold',
                                                                'background-color': '#F9F9F9', 'font-family': 'Verdana', 'font-size': '13px'}),
                    dbc.ListGroupItem(id = 'som-ytd-text', style = {'background-color': '#01529C', 'width': '244px',
                                                                 'font-weight': 'bold', 'color': 'white',
                                                                    'font-size': '11px', 'border-radius': '5px',
                                                                    'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'},
                                      ),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '1px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= 'som-next-text', style = {'background-color': '#01529C', 'width': '244px',
                                                               'color': 'white',
                                                               'font-weight': 'bold',
                                                               'font-size': '11px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '1px', 'background-color': '#F9F9F9', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(id= 'som-next-year-text', style = {'background-color': '#01529C', 'width': '244px',
                                                               'color': 'white',
                                                               'font-weight': 'bold',
                                                               'font-size': '11px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'}),

                ], horizontal=True),
                html.Br(),
                dbc.ListGroup([
                    dbc.ListGroupItem(children= '', style = {'width': '40px', 'font-weight': 'bold', 'color': '#F9F9F9',
                                                                    'font-color': 'white', 'border': 'none',
                                                                    'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '13px'}),
                    dbc.ListGroupItem(id = 'sales-ytd', style = {'font-weight': 'bold', 'border': 'none', 'text-align': 'right',
                                                                         'width': '130px', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '14px'}),
                    dbc.ListGroupItem(id = 'sales-ytd-change'),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '1px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= 'sales-next', style = {'font-weight': 'bold', 'width': '130px', 'text-align': 'right',
                                                                          'border': 'none', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '14px'}),
                    dbc.ListGroupItem(id= 'sales-next-change'),
                    dbc.ListGroupItem(style = {'border': 'none','width': '1px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= 'sales-next-year', style = {'font-weight': 'bold', 'border': 'none', 'text-align': 'right', 'width': '130px',
                                                                           'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '14px'}),
                    dbc.ListGroupItem(id= 'sales-next-year-change'),
                ], horizontal=True),

                dbc.ListGroup([
                    dbc.ListGroupItem(children= '', style= {'width': '40px', 'border': 'none', 'font-weight': 'bold',
                                                                'background-color': '#F9F9F9', 'font-family': 'Verdana', 'font-size': '13px'}),
                    dbc.ListGroupItem(id = 'sales-ytd-text', style = {'background-color': '#01529C', 'width': '244px',
                                                                 'font-weight': 'bold', 'color': 'white',
                                                                    'font-size': '11px', 'border-radius': '5px',
                                                                    'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'},
                                      ),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '1px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= 'sales-next-text', style = {'background-color': '#01529C', 'width': '244px',
                                                               'color': 'white',
                                                               'font-weight': 'bold',
                                                               'font-size': '11px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '1px', 'background-color': '#F9F9F9', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(id= 'sales-next-year-text', style = {'background-color': '#01529C', 'width': '244px',
                                                               'color': 'white',
                                                               'font-weight': 'bold',
                                                               'font-size': '11px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'}),

                ], horizontal=True),


            ], style= {'border': 'none', 'background-color': '#F9F9F9'}

            ),
        ], width = 8)
    ]),

#############################################################

    dtc.Carousel([
            dcc.Graph(id='som-graph', style = {'border-right': '1px solid grey', 'border-left': '1px solid grey', 
                                                'margin-top': '5px', 'margin-bottom': '10px'}),
            dcc.Graph(id='sales-graph', style = {'border-right': '1px solid grey', 'border-left': '1px solid grey',
                                                'margin-top': '5px', 'margin-bottom': '10px'}),
            dcc.Graph(id='units-graph', style = {'border-right': '1px solid grey', 'border-left': '1px solid grey',
                                                'margin-top': '5px', 'margin-bottom': '10px'})
        ],
                    slides_to_scroll=1,
                    swipe_to_slide=True,
                    autoplay=False,
                    variable_width=False,
                    infinite=True,
                    arrows= True,
                    slides_to_show = 2,
                    center_padding = '500px',
                    style = {'padding-left': '20px', 'padding-right': '20px', 
                                'border-top': '2px solid grey', 'border-bottom': '2px solid grey', 'margin': '0px'},
                    dots=True
                ),
    dbc.Row([
            dbc.Col([
                html.Br()
            ], width=12)
        ], style= {'margin': '0px'}),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children= 'SOM Simulation Table', style= {'width': '100%', 'color': '#1D4693',
                                                                                      'background-color': '#E8E8E8', 'border': 'none',
                                                                                     'font-weight': 'bold', 'font-size': '100%', 'font-family': 'Verdana'}),
                ])

            ], style= {'border': 'none'}

            ),
        ], width= 12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Button(
            "Export", id="export-button", n_clicks=0, style= {'width': '50px', 'height': '23px',
                                                            'border-radius': '0px', 'font-size': '11px',
                                                            'text-align': 'center', 'color': 'buttontext',
                                                            'background-color': 'buttonface', 'font-family': 'Verdana',
                                                            'padding': '1px 6px', 'border-width': '2px',
                                                            'border-style': 'outset', 'border-color': 'buttonborder',
                                                            'border-image': 'initial', 'display': 'inline-block',
                                                            'margin-bottom': '2px'}
        ),
            dcc.Download(id="download-dataframe-csv"),

            dbc.Button(
            "Reset", id="reset-button", n_clicks=0, style= {'width': '50px', 'height': '23px',
                                                            'border-radius': '0px', 'font-size': '11px',
                                                            'text-align': 'center', 'color': 'buttontext',
                                                            'background-color': 'buttonface', 'font-family': 'Verdana',
                                                            'padding': '1px 6px', 'border-width': '2px',
                                                            'border-style': 'outset', 'border-color': 'buttonborder',
                                                            'border-image': 'initial', 'display': 'inline-block',
                                                            'margin-bottom': '2px', 'margin-top': '1px'}
        )
        ], width= 2),
        
        dbc.Col([
            dbc.Card(style = {'width': '12px', 'height': '12px', 'margin-top': '8px',
                                'background-color': '#01529C', 'border': 'none', 'border-radius': '0px'}),
        ], width= 1.5, style= {'margin-left': '20%'}),
        dbc.Col([
            dbc.Card("Fixed Columns", style = {'width': '120px', 'height': '12px', 'margin-top': '3px', 'background-color': '#F9F9F9',
                                'color': 'black', 'border': 'none', 'font-family': 'Verdana', 'font-size': '13px'}),
        ], width= 2, style= {'width': '100px'}),
        dbc.Col([
            dbc.Card(style = {'width': '12px', 'height': '12px', 'margin-top': '8px',
                                'background-color': '#F07836', 'border': 'none', 'border-radius': '0px'}),
        ], width= 1.5),
        dbc.Col([
            dbc.Card("Simulation Columns", style = {'height': '12px', 'margin-top': '3px', 'background-color': '#F9F9F9',
                                'color': 'black', 'border': 'none', 'font-family': 'Verdana', 'font-size': '13px'}),
        ], width= 2),
    ]),

    dbc.Row([
        dbc.Col([
            dash_table.DataTable(id='table-editing-simple',
                                style_header={ 'border': '1px solid white', 'whiteSpace':'normal', 'color': 'white',
                                              'font-weight': 'bold', 'backgroundColor': '#01529C', 'font-family': 'Verdana',
                                              'font-size':'10px'},
                                style_cell={ 'border': '1px solid grey', 'minWidth': 85, 'maxWidth': 120,
                                            'background-color': '#F9F9F9', 'font-family': 'Verdana', 'font-size':'10px'},
                                style_table={'overflowX': 'auto', 'height': '300px', 'overflowY': 'auto'},
                                virtualization=True,
                                fixed_rows={'headers': True},
                                style_header_conditional=[{
                                    'if': {'column_editable': True},
                                    'backgroundColor': '#F17836',
                                    'color': 'white'
                                    }]
                                )
        ], width=12, style= {'font-size': '11px'}, align="center")
    ]),

    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ], style= {'border-top': '1px solid grey', 'margin': '0px'}),

    dbc.Row([
        dbc.Col([
            dbc.RadioItems(
                options=[
                    {"label": "Monthly", "value": "Monthly"},
                    {"label": "Yearly", "value": "Yearly"},
                ], value= 'Monthly',
                id="time-selector",
                labelClassName="date-group-labels",
                labelCheckedClassName="date-group-labels-checked",
                className="date-group-items",
                inline=True
        ),
            dcc.Graph(id= 'dist-plot')
        ], width=6, style = {'border-right': '2px solid grey', 'margin-top': '5px', 'margin-bottom': '10px'}),
        dbc.Col([
            dcc.Graph(id='sensitivity-graph'),
            html.P("Above chart will provide users with the primary variables which will affect Sales & SOM in the data table above. Features having importance greater than 0.1% can be seen here.",
                   style= {'height': '20px', 'font-family': 'Verdana', 'font-size': '13px', 'margin-bottom': '5px'})
        ], width=6)
    ], style= {'border-top': '2px solid grey', 'margin': '0px', 'border-bottom': '2px solid grey'}),

    dbc.Row([
        dbc.Col([
            dash_table.DataTable(id='table-editing-simple-second',
                                style_header={ 'border': '1px solid black' },
                                style_cell={ 'border': '1px solid grey' },
                                style_table={'overflowX': 'auto'})
        ], width=12, style= {'font-size': '11px', 'display': 'none'}, align="center")
    ]),

    dbc.Row([
        dbc.Col([
            dash_table.DataTable(id='table-editing-simple-third',
                                style_header={ 'border': '1px solid black' },
                                style_cell={ 'border': '1px solid grey' },
                                style_table={'overflowX': 'auto'})
        ], width=12, style= {'font-size': '11px'}, align="center")
    ]),

    dbc.Row([
        dbc.Col([
            dash_table.DataTable(id='som-metric-table',
                                style_header={ 'border': '1px solid black' },
                                style_cell={ 'border': '1px solid grey' },
                                style_table={'overflowX': 'auto'})
        ], width=3, style= {'font-size': '11px', 'display': 'none'}, align="center"),
        dbc.Col([
            dash_table.DataTable(id='sales-metric-table',
                                style_header={ 'border': '1px solid black' },
                                style_cell={ 'border': '1px solid grey' },
                                style_table={'overflowX': 'auto'})
        ], width=3, style= {'font-size': '11px', 'display': 'none'}, align="center")
    ]),


], fluid = True, style = {'background-color': '#F9F9F9'})


## Callbacks functions to make the dash app interactive

# Filters --> source location for images below filter section
@callback(
    Output('manufacturer-logo', 'src'),
    Output('channel-logo', 'src'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value')
)
def logo_output(mfg, cnl):
    """This is a function that returns the source location of the image for selected manufacturer and channel 

    Args:
        mfg (string): Manufacturer value from Manufacturer filter
        cnl (string): Channel value from Channel filter

    Returns:
        tuple: image locations
    """
    mfg_src = f'./assets/{mfg}.png'
    cnl_src = f'./assets/{cnl}.png'
    return mfg_src, cnl_src

# Filters --> Feature importance plot
@callback(
    Output('sensitivity-graph', 'figure'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value')
)
def sensitivity_plot(mfg, cnl):
    """This function returns feature importance plot from df_feat_imp dataframe

    Args:
        mfg (string): Manufacturer value from Manufacturer filter
        cnl (string): Channel value from Channel filter

    Returns:
        figure: feature importance plot
    """
    global df_feat_imp

    if cnl != 'TOTAL':
        # sensitivity plots
        shap_importance = df_feat_imp[(df_feat_imp['Manufacturer']==mfg)&(df_feat_imp['Channel']==cnl)]
        # print(shap_importance['Feature'].values)
        # print(shap_importance)
        # shap_importance = shap_importance[abs(shap_importance['feature_importance_vals'])>0.01]
        fig_sensitivity = make_subplots()
        fig_sensitivity = fig_sensitivity.add_trace(go.Bar(
                x=shap_importance['feature_importance_vals'],
                y=shap_importance['Feature'],
                orientation='h',
                text=np.round(shap_importance['feature_importance_vals'],3),
                marker=dict(color =shap_importance['Color'])))

        fig_sensitivity.update_layout(paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                        xaxis_title = 'Importance',
                        yaxis_tickangle=-45,
                        title= f'{mfg} X {cnl} Feature Importance'
                        )
    else:
        fig_sensitivity = go.Figure()
        fig_sensitivity.update_layout(
            xaxis =  { "visible": False },
            yaxis = { "visible": False },
            annotations = [
                {
                    "text": "Please Select Channel Except TOTAL",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28
                    }
                }
            ]
        )
    return fig_sensitivity

# Filters and number of clicks on reset button --> inputs for datatable
@callback(
    Output('table-editing-simple', 'data'),
    Output('table-editing-simple', 'columns'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value'),
    Input('reset-button', 'n_clicks'))
def actualize_db(mfg, cnl, n_clicks):
    """This function returns historical and next 18 months data where users can change values in simulated columns to 
    view the changes in both above graphs

    Args:
        mfg (string): Manufacturer value from Manufacturer filter
        cnl (string): Channel value from Channel filter
        n_clicks (integer): number of clicks on reset button

    Returns:
        tuple: dataframe (below the SOM and Sales graph) records and columns
    """
    global display_cols, df_sales_pred, df_all, model_features_data
    if cnl != 'TOTAL':
        #print(cnl, mfg)
        df_slice = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)]
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        
        df_slice['Month'] = df_slice['Month'].astype(int)
        # df_slice = generate_comp_features_2(df_slice)
        df_slice = df_slice.merge(df_sales_pred[['Manufacturer', 'Channel', 'Date', 'units_pred', 'sales_pred', 'SOM_hat']],
                                on = ['Manufacturer', 'Channel', 'Date'])
        df_slice = df_slice.rename(columns= {'units_pred': 'Predicted Units', 'sales_pred': 'Predicted Sales', 
                                             'SOM_hat': 'Predicted SOM'})
        df_slice['Date'] = df_slice['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        selected_cols = model_features_data[(model_features_data['combination']==f'{cnl}_{mfg}')]['features'].values[0]
        # fil_sel_cols = [i.split(' Ratio')[0] for i in selected_cols if 'Ratio' in str(i)]
        fil_sel_cols = [i for i in selected_cols if i not in seasonality_cols]
        avg_cols = [i for i in fil_sel_cols if 'PPU' in str(i)]
        if avg_cols and f'{mfg} PPU' not in avg_cols:
            avg_cols = avg_cols + [f'{mfg} PPU']
        tdp_cols = [i for i in fil_sel_cols if 'TDP' in str(i)]
        if tdp_cols and f'{mfg} TDP' not in tdp_cols:
            tdp_cols = tdp_cols + [f'{mfg} TDP']
        # selected_cols = [i for i in selected_cols if ('Ratio' not in str(i)) and (str(i).isdigit() == False)]
        
        selected_cols = list(set(avg_cols + tdp_cols))
        
        List2 = selected_cols
        selected_cols2 = sorted(List2, key=sort_fun)
        prediction_cols = ['Predicted Units', 'Predicted Sales', 'Predicted SOM']
        prediction_cols = [
            {'name': i, 'id': i, 'editable': False} for i in prediction_cols
        ]
        list_cols = [
            {'name': i, 'id': i, 'editable': True} for i in selected_cols2
        ]
        list_cols = display_cols + prediction_cols + list_cols
        cols_to_display = ['Manufacturer', 'Channel', 'Date', 'Units', 'Predicted Units', 'Sales', 'Predicted Sales', 'SOM', 'Predicted SOM', 'Month'] + selected_cols
        df_slice = df_slice.tail(26)[cols_to_display]
        
    else:
        df_slice = df_all[(df_all['Manufacturer']==mfg)]
        df_slice['Date'] = df_slice['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_slice = df_slice[['Manufacturer', 'Channel', 'Date', 'Units', 'Sales',  'SOM', 'units_pred', 'sales_pred', 'SOM_hat']]
        df_slice = df_slice.rename(columns= {'units_pred': 'Predicted Units',
                                              'sales_pred': 'Predicted Sales',
                                              'SOM_hat': 'Predicted SOM'})
        df_slice = df_slice.round(decimals= 2)
        df_slice = df_slice.tail(26)
        list_cols = [
            {'name': i, 'id': i, 'editable': False} for i in df_slice.columns
        ]

    if n_clicks > 0:
        if cnl != 'TOTAL':
            df_slice = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)]
            df_slice['Date'] = pd.to_datetime(df_slice['Date'])
            df_slice['Month'] = df_slice['Month'].astype(int)
            # df_slice = generate_comp_features_2(df_slice)
            df_slice = df_slice.merge(df_sales_pred[['Manufacturer', 'Channel', 'Date', 'units_pred', 'sales_pred', 'SOM_hat']],
                                    on = ['Manufacturer', 'Channel', 'Date'])
            df_slice = df_slice.rename(columns= {'units_pred': 'Predicted Units', 'sales_pred': 'Predicted Sales', 'SOM_hat': 'Predicted SOM'})
            df_slice['Date'] = df_slice['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            selected_cols = model_features_data[(model_features_data['combination']==f'{cnl}_{mfg}')]['features'].values[0]
            # fil_sel_cols = [i.split(' Ratio')[0] for i in selected_cols if 'Ratio' in str(i)]
            fil_sel_cols = [i for i in selected_cols if i not in seasonality_cols]
            avg_cols = [i for i in fil_sel_cols if 'PPU' in str(i)]
            if avg_cols and f'{mfg} PPU' not in avg_cols:
                avg_cols = avg_cols + [f'{mfg} PPU']
            tdp_cols = [i for i in fil_sel_cols if 'TDP' in str(i)]
            if tdp_cols and f'{mfg} TDP' not in tdp_cols:
                tdp_cols = tdp_cols + [f'{mfg} TDP']
            # selected_cols = [i for i in selected_cols if ('Ratio' not in str(i)) and (str(i).isdigit() == False)]
            
            selected_cols = list(set(avg_cols + tdp_cols))
            List2 = selected_cols
            selected_cols2 = sorted(List2, key=sort_fun)
            prediction_cols = ['Predicted Units', 'Predicted Sales', 'Predicted SOM']
            prediction_cols = [
                {'name': i, 'id': i, 'editable': False} for i in prediction_cols
            ]
            list_cols = [
                {'name': i, 'id': i, 'editable': True} for i in selected_cols2
            ]
            list_cols = display_cols + prediction_cols + list_cols
            cols_to_display = ['Manufacturer', 'Channel', 'Date', 'Units', 'Sales', 'Predicted Units', 'Predicted Sales', 'SOM', 'Predicted SOM', 'Month'] + selected_cols
            df_slice = df_slice.tail(26)[cols_to_display]

        else:
            df_slice = df_all[(df_all['Manufacturer']==mfg)]
            df_slice['Date'] = df_slice['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            df_slice = df_slice[['Manufacturer', 'Channel', 'Date', 'Units', 'Sales',  'SOM', 'units_pred', 'sales_pred', 'SOM_hat']]
            df_slice = df_slice.rename(columns= {'units_pred': 'Predicted Units',
                                                'sales_pred': 'Predicted Sales',
                                                'SOM_hat': 'Predicted SOM'})
            df_slice = df_slice.round(decimals= 2)
            df_slice = df_slice.tail(26)
            list_cols = [
                {'name': i, 'id': i, 'editable': False} for i in df_slice.columns
            ]

    return df_slice.to_dict('records'), list_cols

# Above datatable --> downloadable dataframe format
@callback(
    Output("download-dataframe-csv", "data"),
    Output("export-button", "n_clicks"),
    Input("export-button", "n_clicks"),
    Input('table-editing-simple', 'data'),
    Input('table-editing-simple', 'columns'),
    prevent_initial_call=True,
)
def download_data(n_clicks, rows, columns):
    """This function takes dataframe generated from above function and send it to a downloadable csv format whenever
    we click on export button

    Args:
        n_clicks (integer): number of clicks on export button
        rows (Dataframe or dict): Records
        columns (dict): Dictionary of columns having name and id as a key

    Returns:
        dataframe: Exportable csv format dataframe
        integer: reset the number of clicks on export button to 0
    """
    if n_clicks > 0:
        df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        n_clicks = 0
        return dcc.send_data_frame(df.to_csv, "simulation_data.csv"), n_clicks
    else:
        n_clicks = 0
        return None, n_clicks

# Datatable from actualize_db function --> dataframe with engineered features on editaed data
@callback(
    Output('table-editing-simple-second', 'data'),
    Output('table-editing-simple-second', 'columns'),
    Input('table-editing-simple', 'data')
)
def upDate_columns(data):
    """This function return a dataframe by taking editable dataframe as an input and generate 
    required features with those editabel values

    Args:
        data (dataframe): Updated datframe with engineered features

    Returns:
        tuple: dataframe records and columns
    """
    global display_cols, df_comp, model_features_data
    df_slice = pd.DataFrame(data)
    Channel = df_slice.Channel.unique()[0]
    if Channel != 'TOTAL':
        df_slice = df_slice.drop(columns = ['Predicted Units', 'Predicted Sales', 'Predicted SOM'])
        mfg, cnl = df_slice['Manufacturer'].unique()[0], df_slice['Channel'].unique()[0]
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        df_slice = df_slice.sort_values(by = 'Date')
        df_slice['Month'] = df_slice['Month'].astype(int)
        df_prev = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)]
        df_prev['Date'] = pd.to_datetime(df_prev['Date'])
        df_prev = df_prev.sort_values(by = 'Date')
        selected_cols = model_features_data[(model_features_data['combination']==f'{cnl}_{mfg}')]['features'].values[0]
        # fil_sel_cols = [i.split(' Ratio')[0] for i in selected_cols if 'Ratio' in str(i)]
        fil_sel_cols = [i for i in selected_cols if i not in seasonality_cols]
        avg_cols = [i for i in fil_sel_cols if 'PPU' in str(i)]
        if avg_cols and f'{mfg} PPU' not in avg_cols:
            avg_cols = avg_cols + [f'{mfg} PPU']
        tdp_cols = [i for i in fil_sel_cols if 'TDP' in str(i)]
        if tdp_cols and f'{mfg} TDP' not in tdp_cols:
            tdp_cols = tdp_cols + [f'{mfg} TDP']
        # selected_cols = [i for i in selected_cols if ('Ratio' not in str(i)) and (str(i).isdigit() == False)]
        
        selected_cols = list(set(avg_cols + tdp_cols))
        cols_to_display = ['Manufacturer', 'Channel', 'Date', 'Units', 'Sales', 'SOM', 'PPU', 'TDP', 'Month'] + selected_cols
        df_prev = df_prev.drop(columns = cols_to_display)
        df_prev = df_prev.tail(26)
        df_slice = pd.concat([df_slice, df_prev.set_index(df_slice.index)], axis=1)
        avg_cols = avg_price_features
        tdp_cols = tdp_features

        # for col in avg_cols:
        #     df_slice[f'{col} Ratio'] = df_slice[col].astype(float).div(df_slice[f'{mfg} PPU'].astype(float))

        # for col in tdp_cols:
        #     df_slice[f'{col} Ratio'] = df_slice[col].astype(float).div(df_slice[f'{mfg} TDP'].astype(float))

    else:
        df_slice = df_slice
    
    return df_slice.to_dict('records'), [{"name": i, "id": i} for i in df_slice.columns]

# Filters and editable datatable --> cell styling in the editabel datatable
@callback(
    Output('table-editing-simple', 'style_data_conditional'),
    Input('table-editing-simple', 'data'),
    Input('table-editing-simple', 'columns'),
    Input('chosen-channel', 'value'),
    Input('chosen-manufacturer', 'value')
)
def style_data_table(rows, columns, cnl, mfg):
    """This function defines the conditional formatting for the datable based on any changes in the existing datatable. 
    The cell color changes if we edit any values in that cell in datatable

    Args:
        rows (Dataframe or dict): Records
        columns (dict): Dictionary of columns having name and id as a key
        mfg (string): Manufacturer value from Manufacturer filter
        cnl (string): Channel value from Channel filter

    Returns:
        tuple: tuple containing locations and background color of cells
    """
    global display_cols, df_sales_pred
    if cnl != 'TOTAL':
        cols = [c['name'] for c in columns]
        cols = cols[8:]
        data = pd.DataFrame(rows, columns=cols)
        # print(data.shape)
        index = data.index.tolist()

        df_slice = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)]
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        df_slice = df_slice.sort_values(by='Date')
        df_slice['Month'] = df_slice['Month'].astype(int)
        # df_slice['Month'] = df_slice['Date'].dt.month
        # df_slice = generate_comp_features_2(df_slice)
        df_slice = df_slice.merge(df_sales_pred[['Manufacturer', 'Channel', 'Date', 'units_pred', 'sales_pred', 'SOM_hat']],
                                on = ['Manufacturer', 'Channel', 'Date'])
        df_slice = df_slice.rename(columns= {'units_pred': 'Predicted Units', 'sales_pred': 'Predicted Sales', 
                                             'SOM_hat': 'Predicted SOM'})
        df_slice['Date'] = df_slice['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))

        data_previous = df_slice.tail(26)[cols]
        data_previous.index = index
        style_data = diff_dashtable(data, data_previous)
        style_data_format = [{'if': {'row_index': row['row_index'], 'column_id': row['column_id']},
                              'background-color': 'rgba(240,120,54, 0.2)'} for row in style_data]
    else:
        style_data_format = []
    return style_data_format

# Filters and dataframe having engineered features --> graphs and kpi values
@callback(
    Output('som-graph', 'figure'),
    Output('sales-graph', 'figure'),
    Output('units-graph', 'figure'),
    Output('som-ytd', 'children'),
    Output('som-ytd-text', 'children'),
    Output('som-ytd-change', 'children'),
    Output('som-ytd-change', 'style'),
    Output('sales-ytd', 'children'),
    Output('sales-ytd-text', 'children'),
    Output('sales-ytd-change', 'children'),
    Output('sales-ytd-change', 'style'),
    Output('som-next', 'children'),
    Output('som-next-text', 'children'),
    Output('som-next-change', 'children'),
    Output('som-next-change', 'style'),
    Output('som-next-year', 'children'),
    Output('som-next-year-text', 'children'),
    Output('som-next-year-change', 'children'),
    Output('som-next-year-change', 'style'),
    Output('sales-next', 'children'),
    Output('sales-next-text', 'children'),
    Output('sales-next-change', 'children'),
    Output('sales-next-change', 'style'),
    Output('sales-next-year', 'children'),
    Output('sales-next-year-text', 'children'),
    Output('sales-next-year-change', 'children'),
    Output('sales-next-year-change', 'style'),
    Output('dist-plot', 'figure'),
    Input('table-editing-simple-second', 'data'),
    Input('table-editing-simple-second', 'columns'),
    Input('chosen-channel', 'value'),
    Input('chosen-manufacturer', 'value'),
    Input('time-selector', 'value'))

def display_output(rows, columns, chnl, mftr, radio_value):
    """This function takes the dataframe having changed values and model will generate predictions on that. 
    Based on forecasted sales, SOM and Sales plots, Source of growth plots, and kpi values are generated

    Args:
        rows (Dataframe or dict): Records
        columns (dict): Dictionary of columns having name and id as a key
        chnl (string): Channel value from Channel filter
        mftr (string): Manufacturer value from Manufacturer filter
        radio_value (string): Monthly or Yearly value

    Returns:
        tuple: tuple containing graphs and kpi values
    """
    global df_comp, regression_features, df_sales_pred, max_date, df_all, model_features_data, df_coef, realistic_df
    
    styles = {'font-weight': 'bold', 'border': 'black', 'text-align': 'left',
              'width': '114px', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '12px'}
    if chnl != 'TOTAL':

        df_rest = df_sales_pred.copy()
        df_slice = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        df_slice['Month'] = df_slice['Month'].astype(int)
        assert df_slice['Manufacturer'].nunique()==1
        assert df_slice['Channel'].nunique()==1

        mfg = df_slice['Manufacturer'].unique()[0]
        cnl = df_slice['Channel'].unique()[0]

        df_slice_whole = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)].iloc[:-26]
        df_slice_act = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)].sort_values(by='Date').tail(26)

        ################  
        realistic_df_1 = realistic_df[(realistic_df['Channel']==cnl) & (realistic_df['Manufacturer']==mfg)]
        max_ppu = realistic_df_1['avg_ppu'].max()
        max_sales = realistic_df_1['avg_sales'].max()
        test = pd.DataFrame()
        test['Date'] = df_slice['Date']
        test['changed'] = df_slice[f'{mfg} PPU'].astype(float)
        test['actual'] = df_slice_act[f'{mfg} PPU'].values.astype(float)

        test['Manufacturer'] = mfg
        test['Channel'] = cnl
        test = test[test['changed']>=max_ppu]
        dates = test.loc[(test['actual']!=test['changed'])]['Date'].values

        df_slice_whole['Date'] = pd.to_datetime(df_slice_whole['Date'])
        
        # df_slice_whole = generate_comp_features_2(df_slice_whole)
        df_slice_whole = round(df_slice_whole,2)
        
        df_slice_whole['Month'] = df_slice_whole['Month'].astype(int)
        df_slice_whole = df_slice_whole.sort_values(by='Date')
        
        columns = df_slice.columns.tolist()
        df_slice_whole = pd.concat([df_slice_whole[columns], df_slice])
        if cnl == 'TRADITIONAL':
            df_slice_whole = df_slice_whole[df_slice_whole['Date']>'2019-03-01']
        
        df_slice2 = df_slice_whole.copy()
    
        df_rest = df_rest[~((df_rest['Manufacturer']==mfg)&(df_rest['Channel']==cnl))][columns + ['units_pred', 'sales_pred']]
        df_rest['Date'] = pd.to_datetime(df_rest['Date'])
        df_rest['units_pred'] = df_rest['units_pred'].astype(float)
        df_rest['sales_pred'] = df_rest['sales_pred'].astype(float)

        req_data = model_features_data[model_features_data['combination'] == f"{cnl}_{mfg}"].values
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

        sel_feat = features_
        log_features = [i for i in sel_feat]
        
        oh_cols = list(pd.get_dummies(df_slice2['Month']).columns.values)
        encoded_features = pd.get_dummies(df_slice2['Month'])
        df_slice2 = pd.concat([df_slice2, encoded_features],axis=1)
        df_slice2 = df_slice2.sort_values(by='Date')
        coeff = df_coef[(df_coef['Manufacturer']==mfg)&(df_coef['Channel']==cnl)]
        df_slice2[log_features] = np.where(df_slice2[log_features] != 0, df_slice2[log_features].astype(float).apply(np.log10), 0)

        y = final_processing(df_slice2, sel_feat, coeff)
        df_slice_whole['units_pred'] = y
        
        # print(y)
        mean_value = np.mean(y[-18:])
        df_slice_whole['units_pred'] = np.where(df_slice_whole['units_pred'] < 0, mean_value, df_slice_whole['units_pred'])
        # df_slice_whole = fill_nan_with_mean_from_prev_and_next(df_slice_whole, ['units_pred'])
        
        df_slice_whole['sales_pred'] = df_slice_whole['units_pred'] * df_slice_whole[f'{mfg} PPU'].astype(float)

        # check for PPU in df_slice_whole on the changed dates if greater than max ppu then adjust sales , and pick the minimum 
        # print(df_slice_whole.columns)
        for date in dates:
            df_slice_whole_real = df_slice_whole[df_slice_whole['Date']==date]
            pred_sale_num = df_slice_whole_real['sales_pred'].values.astype(float)
            ppu_num = df_slice_whole_real[f'{mfg} PPU'].values.astype(float)
            if ppu_num >= max_ppu:
                pred_sale_num_changed = calculate_sales(ppu_num, max_sales, max_ppu)
                # print(pred_sale_num, pred_sale_num_changed)
                df_slice_whole.loc[df_slice_whole['Date']==date, 'sales_pred'] = pred_sale_num_changed
        
        df_whole = pd.concat([df_slice_whole, df_rest])
        df_whole['total_sales_hat'] = df_whole.groupby(['Date', 'Channel'])['sales_pred'].transform('sum')
        df_whole['SOM_hat'] = df_whole['sales_pred']*100/df_whole['total_sales_hat']

        # Som and sales card outputs
        card_outputs = df_whole[(df_whole['Manufacturer']==mfg)&(df_whole['Channel']==cnl)]
        card_outputs['year'] = card_outputs['Date'].dt.year
        next_month = (max_date + relativedelta(months=1)).strftime('%b-%Y')
        current_year = (max_date + relativedelta(months=1)).year
        som_next_text, sales_next_text = f'Estimated SOM For {next_month}', f'Estimated Sales For {next_month} (10^3 Pesos)'
        som_next_year_text, sales_next_year_text = f'Estimated SOM For Total {current_year}', f'Estimated Sales For Total {current_year} (10^3 Pesos)'
        som_next, sales_next = (np.round(card_outputs[card_outputs['Date']==next_month]['SOM_hat'].values[0], 2),
                                np.int(card_outputs[card_outputs['Date']==next_month]['sales_pred'].values[0]))
        
        ######### % change from previous month som and sales
        som_prev, sales_prev = (np.round(card_outputs[card_outputs['Date']==max_date]['SOM'].values[0], 2),
                                np.int(card_outputs[card_outputs['Date']==max_date]['Sales'].values[0]))
        som_next_change, sales_next_change = som_next - som_prev, (sales_next-sales_prev)*100/sales_prev
        som_next_style, sales_next_style = styles.copy(), styles.copy()
        som_next_style['color'] = '#009849' if som_next_change >= 0 else '#C9002B'
        sales_next_style['color'] = '#009849' if sales_next_change >= 0 else '#C9002B'
        som_next_change = f'▲{som_next_change:.2f} %' if som_next_change >= 0 else  f'▼{abs(som_next_change):.2f} %'
        sales_next_change = f'▲{sales_next_change:.2f} %' if sales_next_change >= 0 else  f'▼{abs(sales_next_change):.2f} %'
        ############################
        som_next, sales_next = f'{som_next} %', '{:,}'.format(sales_next)
        
        sales_next_year = card_outputs[card_outputs['year']==current_year]
        sales_next_year['sales_final'] = np.where(sales_next_year['Date']>max_date, sales_next_year['sales_pred'],
                                                sales_next_year['Sales'])
        sales_next_year['sales_final'] = np.where(sales_next_year['sales_final'].isnull(), sales_next_year['sales_pred'],
                                                sales_next_year['sales_final'])
        sales_next_year = np.int(sales_next_year[sales_next_year['year']==current_year]['sales_final'].sum())
        
        ######### % change from previous year sales
        sales_next_year_style = styles.copy()
        sales_prev_year = card_outputs[card_outputs['year']==current_year-1]
        sales_prev_year['sales_final'] = sales_prev_year['Sales']
        sales_prev_year = np.int(sales_prev_year[sales_prev_year['year']==current_year-1]['sales_final'].sum())
        sales_next_year_change = (sales_next_year - sales_prev_year)*100/sales_prev_year
        sales_next_year_style['color'] = '#009849' if sales_next_year_change >= 0 else '#C9002B'
        sales_next_year_change = f'▲{sales_next_year_change:.2f} %' if sales_next_year_change >= 0 else  f'▼{abs(sales_next_year_change):.2f} %'
        ############################
        sales_next_year = '{:,}'.format(sales_next_year)

        som_next_year = df_whole.copy()
        som_next_year['year'] = som_next_year['Date'].dt.year
        som_next_year = som_next_year[som_next_year['year']==current_year]
        som_next_year['sales_final'] = np.where(som_next_year['Date']>max_date, som_next_year['sales_pred'],
                                                som_next_year['Sales'])
        som_next_year['sales_final'] = np.where(som_next_year['sales_final'].isnull(), som_next_year['sales_pred'],
                                                som_next_year['sales_final'])
        som_next_year = som_next_year.groupby(['Manufacturer', 'Channel'])['sales_final'].sum().reset_index()
        som_next_year['sales_final_hat'] = som_next_year.groupby(['Channel'])['sales_final'].transform('sum')
        som_next_year['SOM_hat'] = som_next_year['sales_final']*100/som_next_year['sales_final_hat']
        som_next_year = np.round(som_next_year[(som_next_year['Manufacturer']==mfg)&(som_next_year['Channel']==cnl)]['SOM_hat'].values[0], 2)
        
        ######### % change from previous year som
        som_next_year_style = styles.copy()
        som_prev_year = df_whole.copy()
        som_prev_year['year'] = som_prev_year['Date'].dt.year
        som_prev_year = som_prev_year[som_prev_year['year']==current_year-1]
        som_prev_year['sales_final'] = som_prev_year['Sales']
        som_prev_year = som_prev_year.groupby(['Manufacturer', 'Channel'])['sales_final'].sum().reset_index()
        som_prev_year['sales_final_hat'] = som_prev_year.groupby(['Channel'])['sales_final'].transform('sum')
        som_prev_year['SOM'] = som_prev_year['sales_final']*100/som_prev_year['sales_final_hat']
        som_prev_year = np.round(som_prev_year[(som_prev_year['Manufacturer']==mfg)&(som_prev_year['Channel']==cnl)]['SOM'].values[0], 2)
        som_next_year_change = som_next_year - som_prev_year
        som_next_year_style['color'] = '#009849' if som_next_year_change >= 0 else '#C9002B'
        som_next_year_change = f'▲{som_next_year_change:.2f} %' if som_next_year_change >= 0 else  f'▼{abs(som_next_year_change):.2f} %'
        ############################
        som_next_year = f'{som_next_year} %'

        # YTD sales and som calculation
        ytd = df_whole.copy()
        ytd['year'] = ytd['Date'].dt.year
        ytd = ytd[(ytd['Date']<= max_date)&(ytd['year']==max_date.year)]
        ytd = ytd.groupby(['Manufacturer', 'Channel'])['Sales'].sum().reset_index()
        ytd['sales_sum'] = ytd.groupby(['Channel'])['Sales'].transform('sum')
        ytd['SOM'] = ytd['Sales']*100/ytd['sales_sum']
        som_ytd = np.round(ytd[(ytd['Manufacturer']==mfg)&(ytd['Channel']==cnl)]['SOM'].values[0], 2)
        som_ytd_text, sales_ytd_text = f'SOM YTD {(max_date).year}', f'Sales YTD {(max_date).year} (10^3 Pesos)'
        sales_ytd = np.int(ytd[(ytd['Manufacturer']==mfg)&(ytd['Channel']==cnl)]['Sales'].values[0])

        ######## % change from previous YTD som and sales
        som_ytd_style, sales_ytd_style = styles.copy(), styles.copy()
        ytd_prev = df_whole.copy()
        ytd_prev['year'] = ytd_prev['Date'].dt.year
        ytd_prev = ytd_prev[(ytd_prev['Date']<= max_date-relativedelta(months=12))&(ytd_prev['year']==max_date.year-1)]
        ytd_prev = ytd_prev.groupby(['Manufacturer', 'Channel'])['Sales'].sum().reset_index()
        ytd_prev['sales_sum'] = ytd_prev.groupby(['Channel'])['Sales'].transform('sum')
        ytd_prev['SOM'] = ytd_prev['Sales']*100/ytd_prev['sales_sum']
        som_ytd_prev = np.round(ytd_prev[(ytd_prev['Manufacturer']==mfg)&(ytd_prev['Channel']==cnl)]['SOM'].values[0], 2)
        som_ytd_change = som_ytd - som_ytd_prev
        som_ytd_style['color'] = '#009849' if som_ytd_change >= 0 else '#C9002B'
        som_ytd_change = f'▲{som_ytd_change:.2f} %' if som_ytd_change >= 0 else  f'▼{abs(som_ytd_change):.2f} %'
        
        sales_ytd_prev = np.int(ytd_prev[(ytd_prev['Manufacturer']==mfg)&(ytd_prev['Channel']==cnl)]['Sales'].values[0])
        sales_ytd_change = (sales_ytd-sales_ytd_prev)*100/sales_ytd_prev
        sales_ytd_style['color'] = '#009849' if sales_ytd_change >= 0 else '#C9002B'
        sales_ytd_change = f'▲{sales_ytd_change:.2f} %' if sales_ytd_change >= 0 else  f'▼{abs(sales_ytd_change):.2f} %'
        ###################
        som_ytd = f'{som_ytd} %'
        sales_ytd = '{:,}'.format(sales_ytd)

        ## Sales and SOM plots
        df_plot = df_whole[(df_whole['Manufacturer']==mfg)&(df_whole['Channel']==cnl)]
        df_plot2 = df_plot[df_plot['Date']<=max_date]
        df_plot['SOM'] = np.where(df_plot['Date']==max_date+ relativedelta(months=1),
                                df_plot['SOM_hat'], df_plot['SOM'])
        df_plot['SOM_hat'] = np.where(df_plot['Date']<=max_date, np.nan, df_plot['SOM_hat'])
        df_plot['Sales'] = np.where(df_plot['Date']==max_date + relativedelta(months=1),
                                    df_plot['sales_pred'], df_plot['Sales'])
        df_plot['sales_pred'] = np.where(df_plot['Date']<=max_date, np.nan, df_plot['sales_pred'])
        df_plot['Units'] = np.where(df_plot['Date']==max_date + relativedelta(months=1),
                                    df_plot['units_pred'], df_plot['Units'])
        df_plot['units_pred'] = np.where(df_plot['Date']<=max_date, np.nan, df_plot['units_pred'])

        df_plot = df_plot.rename(columns= {'SOM_hat': 'Predicted SOM',
                                        'sales_pred': 'Predicted Sales', 
                                        'units_pred': 'Predicted Units'})

        fig_som = go.Figure()
        fig_som.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['SOM'], name= 'SOM', line=dict(color='#015CB4')))
        fig_som.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Predicted SOM'],
                                name= 'Predicted SOM', marker= dict(color= '#C9002B')))
        fig_som.add_vrect(x0=df_plot.tail(future_months)['Date'].min(),
                x1=df_plot.tail(future_months)['Date'].max(),
                line_width=0, fillcolor="blue", opacity=0.1)
        fig_som.update_layout(title= 'Share of Market Trend (Historical & Predicted)', yaxis_title = 'SOM (%)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))

        fig_sales = go.Figure()
        fig_sales.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Sales'], name= 'Sales', line=dict(color='#015CB4')))
        fig_sales.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Predicted Sales'],
                                name= 'Predicted Sales', marker= dict(color= '#C9002B')))
        fig_sales.add_vrect(x0=df_plot.tail(future_months)['Date'].min(),
                x1=df_plot.tail(future_months)['Date'].max(),
                line_width=0, fillcolor="blue", opacity=0.1)
        fig_sales.update_layout(title= 'Market Sales Trend (Historical & Predicted)', yaxis_title = 'Sales (10^3 Pesos)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))
        
        fig_units = go.Figure()
        fig_units.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Units'], name= 'Units', line=dict(color='#015CB4')))
        fig_units.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Predicted Units'],
                                name= 'Predicted Units', marker= dict(color= '#C9002B')))
        fig_units.add_vrect(x0=df_plot.tail(future_months)['Date'].min(),
                x1=df_plot.tail(future_months)['Date'].max(),
                line_width=0, fillcolor="blue", opacity=0.1)
        fig_units.update_layout(title= 'Units Trend (Historical & Predicted)', yaxis_title = 'Units (10^3)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))

        # Distribution (SoG) plots
        df_yearly = df_whole.copy()
        df_yearly = df_yearly.rename(columns= {'units_pred': 'Predicted Units',
                                               'sales_pred': 'Predicted Sales',
                                               'SOM_hat': 'Predicted SOM'})
        if radio_value == 'Yearly':
            # yearly SoG plot
            fig_dist = plot_dist_charts(df_yearly, mfg)
        else:
            # monthly SoG plot
            fig_dist = plot_monthly_dist_chart(df_yearly, mfg, current_year)

    else:
        df_plot = df_all[df_all['Manufacturer']==mftr]
        df_plot['SOM'] = np.where(df_plot['Date']==max_date+ relativedelta(months=1),
                                df_plot['SOM_hat'], df_plot['SOM'])
        df_plot['SOM_hat'] = np.where(df_plot['Date']<=max_date, np.nan, df_plot['SOM_hat'])
        df_plot['Sales'] = np.where(df_plot['Date']==max_date + relativedelta(months=1),
                                    df_plot['sales_pred'], df_plot['Sales'])
        df_plot['Sales'] = np.where(df_plot['Date']>max_date + relativedelta(months=1),
                                    np.nan, df_plot['Sales'])
        df_plot['sales_pred'] = np.where(df_plot['Date']<=max_date, np.nan, df_plot['sales_pred'])
        df_plot['Units'] = np.where(df_plot['Date']==max_date + relativedelta(months=1),
                                    df_plot['units_pred'], df_plot['Units'])
        df_plot['Units'] = np.where(df_plot['Date']>max_date + relativedelta(months=1),
                                    np.nan, df_plot['Units'])
        df_plot['units_pred'] = np.where(df_plot['Date']<=max_date, np.nan, df_plot['units_pred'])

        df_plot = df_plot.rename(columns= {'SOM_hat': 'Predicted SOM',
                                        'sales_pred': 'Predicted Sales', 
                                        'units_pred': 'Predicted Units'})
        df_plot = df_plot[['Manufacturer', 'Channel', 'Date', 'Units', 'Sales',  'SOM', 'Predicted Units', 'Predicted Sales', 'Predicted SOM']]

        fig_som = go.Figure()
        fig_som.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['SOM'], name= 'SOM', line=dict(color='#015CB4')))
        fig_som.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Predicted SOM'],
                                name= 'Predicted SOM', marker= dict(color= '#C9002B')))
        fig_som.add_vrect(x0=df_plot.tail(future_months)['Date'].min(),
                x1=df_plot.tail(future_months)['Date'].max(),
                line_width=0, fillcolor="blue", opacity=0.1)
        fig_som.update_layout(title= 'Share of Market Trend (Historical & Predicted)', yaxis_title = 'SOM (%)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))

        fig_sales = go.Figure()
        fig_sales.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Sales'], name= 'Sales', line=dict(color='#015CB4')))
        fig_sales.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Predicted Sales'],
                                name= 'Predicted Sales', marker= dict(color= '#C9002B')))
        fig_sales.add_vrect(x0=df_plot.tail(future_months)['Date'].min(),
                x1=df_plot.tail(future_months)['Date'].max(),
                line_width=0, fillcolor="blue", opacity=0.1)
        fig_sales.update_layout(title= 'Market Sales Trend (Historical & Predicted)', yaxis_title = 'Sales (10^3 Pesos)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))

        fig_units = go.Figure()
        fig_units.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Units'], name= 'Units', line=dict(color='#015CB4')))
        fig_units.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Predicted Units'],
                                name= 'Predicted Units', marker= dict(color= '#C9002B')))
        fig_units.add_vrect(x0=df_plot.tail(future_months)['Date'].min(),
                x1=df_plot.tail(future_months)['Date'].max(),
                line_width=0, fillcolor="blue", opacity=0.1)
        fig_units.update_layout(title= 'Units Trend (Historical & Predicted)', yaxis_title = 'Units (10^3)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))
        
        # Som and sales card outputs
        df_whole = df_all.copy()
        card_outputs = df_whole[(df_whole['Manufacturer']==mftr)]
        card_outputs['year'] = card_outputs['Date'].dt.year
        next_month = (max_date + relativedelta(months=1)).strftime('%b-%Y')
        current_year = (max_date + relativedelta(months=1)).year
        som_next_text, sales_next_text = f'Estimated SOM For {next_month}', f'Estimated Sales For {next_month} (10^3 Pesos)'
        som_next_year_text, sales_next_year_text = f'Estimated SOM For Total {current_year}', f'Estimated Sales For Total {current_year} (10^3 Pesos)'
        som_next, sales_next = (np.round(card_outputs[card_outputs['Date']==next_month]['SOM_hat'].values[0], 2),
                                np.int(card_outputs[card_outputs['Date']==next_month]['sales_pred'].values[0]))
        
        ######### % change from previous month som and sales
        som_prev, sales_prev = (np.round(card_outputs[card_outputs['Date']==max_date]['SOM'].values[0], 2),
                                np.int(card_outputs[card_outputs['Date']==max_date]['Sales'].values[0]))
        som_next_change, sales_next_change = som_next - som_prev, (sales_next-sales_prev)*100/sales_prev
        som_next_style, sales_next_style = styles.copy(), styles.copy()
        som_next_style['color'] = '#009849' if som_next_change >= 0 else '#C9002B'
        sales_next_style['color'] = '#009849' if sales_next_change >= 0 else '#C9002B'
        som_next_change = f'▲{som_next_change:.2f} %' if som_next_change >= 0 else  f'▼{abs(som_next_change):.2f} %'
        sales_next_change = f'▲{sales_next_change:.2f} %' if sales_next_change >= 0 else  f'▼{abs(sales_next_change):.2f} %'
        ############################
        
        som_next, sales_next = f'{som_next} %', '{:,}'.format(sales_next)
        sales_next_year = card_outputs[card_outputs['year']==current_year]
        sales_next_year['sales_final'] = np.where(sales_next_year['Date']>max_date, sales_next_year['sales_pred'],
                                                sales_next_year['Sales'])
        sales_next_year['sales_final'] = np.where(sales_next_year['sales_final'].isnull(), sales_next_year['sales_pred'],
                                                sales_next_year['sales_final'])
        sales_next_year = np.int(sales_next_year[sales_next_year['year']==current_year]['sales_final'].sum())
        ######### % change from previous year sales
        sales_next_year_style = styles.copy()
        sales_prev_year = card_outputs[card_outputs['year']==current_year-1]
        sales_prev_year['sales_final'] = sales_prev_year['Sales']
        sales_prev_year = np.int(sales_prev_year[sales_prev_year['year']==current_year-1]['sales_final'].sum())
        sales_next_year_change = (sales_next_year - sales_prev_year)*100/sales_prev_year
        sales_next_year_style['color'] = '#009849' if sales_next_year_change >= 0 else '#C9002B'
        sales_next_year_change = f'▲{sales_next_year_change:.2f} %' if sales_next_year_change >= 0 else  f'▼{abs(sales_next_year_change):.2f} %'
        ############################
        sales_next_year = '{:,}'.format(sales_next_year)

        som_next_year = df_whole.copy()
        som_next_year['year'] = som_next_year['Date'].dt.year
        som_next_year = som_next_year[som_next_year['year']==current_year]
        som_next_year['sales_final'] = np.where(som_next_year['Date']>max_date, som_next_year['sales_pred'],
                                                som_next_year['Sales'])
        som_next_year['sales_final'] = np.where(som_next_year['sales_final'].isnull(), som_next_year['sales_pred'],
                                                som_next_year['sales_final'])
        som_next_year = som_next_year.groupby(['Manufacturer', 'Channel'])['sales_final'].sum().reset_index()
        som_next_year['sales_final_hat'] = som_next_year.groupby(['Channel'])['sales_final'].transform('sum')
        som_next_year['SOM_hat'] = som_next_year['sales_final']*100/som_next_year['sales_final_hat']
        som_next_year = np.round(som_next_year[(som_next_year['Manufacturer']==mftr)&(som_next_year['Channel']=='TOTAL')]['SOM_hat'].values[0], 2)
        ######### % change from previous year som
        som_next_year_style = styles.copy()
        som_prev_year = df_whole.copy()
        som_prev_year['year'] = som_prev_year['Date'].dt.year
        som_prev_year = som_prev_year[som_prev_year['year']==current_year-1]
        som_prev_year['sales_final'] = som_prev_year['Sales']
        som_prev_year = som_prev_year.groupby(['Manufacturer', 'Channel'])['sales_final'].sum().reset_index()
        som_prev_year['sales_final_hat'] = som_prev_year.groupby(['Channel'])['sales_final'].transform('sum')
        som_prev_year['SOM'] = som_prev_year['sales_final']*100/som_prev_year['sales_final_hat']
        som_prev_year = np.round(som_prev_year[(som_prev_year['Manufacturer']==mftr)&(som_prev_year['Channel']=='TOTAL')]['SOM'].values[0], 2)
        som_next_year_change = som_next_year - som_prev_year
        som_next_year_style['color'] = '#009849' if som_next_year_change >= 0 else '#C9002B'
        som_next_year_change = f'▲{som_next_year_change:.2f} %' if som_next_year_change >= 0 else  f'▼{abs(som_next_year_change):.2f} %'
        ######################
        som_next_year = f'{som_next_year} %'

        # YTD sales and som calculation
        ytd = df_whole.copy()
        ytd['year'] = ytd['Date'].dt.year
        ytd = ytd[(ytd['Date']<= max_date)&(ytd['year']==max_date.year)]
        ytd = ytd.groupby(['Manufacturer', 'Channel'])['Sales'].sum().reset_index()
        ytd['sales_sum'] = ytd.groupby(['Channel'])['Sales'].transform('sum')
        ytd['SOM'] = ytd['Sales']*100/ytd['sales_sum']
        som_ytd = np.round(ytd[(ytd['Manufacturer']==mftr)&(ytd['Channel']=='TOTAL')]['SOM'].values[0], 2)
        som_ytd_text, sales_ytd_text = f'SOM YTD {(max_date).year}', f'Sales YTD {(max_date).year} (10^3 Pesos)'
        sales_ytd = np.int(ytd[(ytd['Manufacturer']==mftr)&(ytd['Channel']=='TOTAL')]['Sales'].values[0])
        
        ######## % change from previous YTD som and sales
        som_ytd_style, sales_ytd_style = styles.copy(), styles.copy()
        ytd_prev = df_whole.copy()
        ytd_prev['year'] = ytd_prev['Date'].dt.year
        ytd_prev = ytd_prev[(ytd_prev['Date']<= max_date-relativedelta(months=12))&(ytd_prev['year']==max_date.year-1)]
        ytd_prev = ytd_prev.groupby(['Manufacturer', 'Channel'])['Sales'].sum().reset_index()
        ytd_prev['sales_sum'] = ytd_prev.groupby(['Channel'])['Sales'].transform('sum')
        ytd_prev['SOM'] = ytd_prev['Sales']*100/ytd_prev['sales_sum']
        som_ytd_prev = np.round(ytd_prev[(ytd_prev['Manufacturer']==mftr)&(ytd_prev['Channel']=='TOTAL')]['SOM'].values[0], 2)
        som_ytd_change = som_ytd - som_ytd_prev
        som_ytd_style['color'] = '#009849' if som_ytd_change >= 0 else '#C9002B'
        som_ytd_change = f'▲{som_ytd_change:.2f} %' if som_ytd_change >= 0 else  f'▼{abs(som_ytd_change):.2f} %'
        
        sales_ytd_prev = np.int(ytd_prev[(ytd_prev['Manufacturer']==mftr)&(ytd_prev['Channel']=='TOTAL')]['Sales'].values[0])
        sales_ytd_change = (sales_ytd-sales_ytd_prev)*100/sales_ytd_prev
        sales_ytd_style['color'] = '#009849' if sales_ytd_change >= 0 else '#C9002B'
        sales_ytd_change = f'▲{sales_ytd_change:.2f} %' if sales_ytd_change >= 0 else  f'▼{abs(sales_ytd_change):.2f} %'
        ###################
        som_ytd = f'{som_ytd} %'
        sales_ytd = '{:,}'.format(sales_ytd)

        # distribution charts
        fig_dist = go.Figure()
        fig_dist.update_layout(
            xaxis =  { "visible": False },
            yaxis = { "visible": False },
            annotations = [
                {
                    "text": "Please Select Channel Except TOTAL",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28
                    }
                }
            ]
        )


    return (fig_som, fig_sales, fig_units,
           som_ytd, som_ytd_text, som_ytd_change, som_ytd_style, sales_ytd, sales_ytd_text, sales_ytd_change, sales_ytd_style, som_next, som_next_text, som_next_change, som_next_style, som_next_year,
            som_next_year_text, som_next_year_change, som_next_year_style, sales_next, sales_next_text, sales_next_change, sales_next_style, sales_next_year, sales_next_year_text, 
            sales_next_year_change, sales_next_year_style, fig_dist
            )
