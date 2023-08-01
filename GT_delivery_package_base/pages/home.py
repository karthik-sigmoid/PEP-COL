import dash
import dash_table
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Registering Home as page in the app
dash.register_page(__name__)

# page contents for the dash app
layout = dbc.Container([
    dbc.Row([
        dbc.ButtonGroup(
        [
            dbc.Button("Home", id= 'navigation-page',
                       style = {'width': '3in', 'background-color': '#01529C', 'color': 'white', 
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       className="ml-auto"),
            dbc.Button("Baseline Prediction", id= 'first-page',
                       style = {'width': '3in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white', 
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/Simulation',
                       className="ml-auto"),
            dbc.Button("Prediction Accuracy", className="ml-auto", id= 'second-page',
                       style = {'width': '3in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white', 
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/ErrorMetrics'
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
            html.Br()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children= 'Navigation', style= {'width': '100%', 'color': '#1D4693',
                                                                      'font-family': 'Verdana',
                                                                      'background-color': '#E8E8E8', 'border': 'none', 
                                                                      'font-weight': 'bold', 'font-size': '120%'}),
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
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children='Page 1- Baseline Prediction', style= {'width': '100%', 'color': 'black',
                                                                  'background-color': '#D8E5F2', 'border': 'none', 'height': '40px', 
                                                                  'font-weight': 'bold', 'font-size': '100%', 'font-family': 'Verdana' }),
                    dbc.ListGroupItem(children= '', style= {'width': '100%', 'background-color': '#D8E5F2', 
                                                            'border': 'none', 'height': '3px'}),
                    dbc.ListGroupItem(children= "Users should visit this page to understand historical movements of Sales & SOM across manufacturers and channels, and look at model's prediction for the next 18 months", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 'font-family': 'Verdana',
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold'}),
                    dbc.ListGroupItem(children= '', style= {'width': '100%', 'background-color': '#D8E5F2', 
                                                            'border': 'none', 'height': '1px'}),
                    dbc.ListGroupItem(children= "• Select Manufacturer & Channel on the top of the page", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• Segment 1: Towards the right of the filters, user can observe summary of YTD, predicted next month & predicted Year end SOM/Sales Values", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• Segment 2: User can observe two line charts explaining the historical and predicted movement os Sales/SOM", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• Segment 3: User can observe a simulation table having historic and predicted sales along with features affecting the predictions", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• Segment 4: User can observe YoY/MoM SOG(Source of Growth) SOM and feature importance plots", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '50px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    
                ])
                
            ], style= {'border': 'none'}
                
            ),
        ], width= 12),
                
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children='Page 2- Prediction Accuracy', style= {'width': '100%', 'color': 'black',
                                                                  'background-color': '#D8E5F2', 'border': 'none', 'height': '40px', 
                                                                  'font-weight': 'bold', 'font-size': '100%', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= '', style= {'width': '100%', 'background-color': '#D8E5F2', 
                                                            'border': 'none', 'height': '3px'}),
                    dbc.ListGroupItem(children= "Users should visit this page to check the prediction model's statistical performance (for Sales & SOM Prediction) across manufacturers and channels", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= '', style= {'width': '100%', 'background-color': '#D8E5F2', 
                                                            'border': 'none', 'height': '1px'}),
                    dbc.ListGroupItem(children= "• Select Manufacturer, Channel & Error parameter on the top of the page", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• Segment 1:  Model Training performance for last month, last 6 months, and last 12 months and same plot can be observed", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• Segment 2: Model Performance Matrix can be observed, it shows model trained incrementally on monthly basis and the corresponding results", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '50px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    
                ])
                
            ], style= {'border': 'none'}
                
            ),
        ], width= 12),
                
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children='Page 3- Scenario Management', style= {'width': '100%', 'color': 'black',
                                                                  'background-color': '#D8E5F2', 'border': 'none', 'height': '40px', 
                                                                  'font-weight': 'bold', 'font-size': '100%', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= '', style= {'width': '100%', 'background-color': '#D8E5F2', 
                                                            'border': 'none', 'height': '3px'}),
                    dbc.ListGroupItem(children= "Users should utilize this capability to run What-If analysis and simulate real-world scenarios. Users can also compare the user-generated scenario with the baseline scenario and download results through the 1-Click download functionality", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '50px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= '', style= {'width': '100%', 'background-color': '#D8E5F2', 
                                                            'border': 'none', 'height': '1px'}),
                    dbc.ListGroupItem(children= '• Download the template by clicking on "Download Template" button at the top', 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• Open the template file and make changes to PPU and TDP values", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• Upload the modified template file to view the simulation results", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '50px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    
                ])
                
            ], style= {'border': 'none'}
                
            ),
        ], width= 12),
                
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children='The Glossary', style= {'width': '100%', 'color': 'black',
                                                                  'background-color': '#D8E5F2', 'border': 'none', 'height': '40px', 
                                                                  'font-weight': 'bold', 'font-size': '100%', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= '', style= {'width': '100%', 'background-color': '#D8E5F2', 
                                                            'border': 'none', 'height': '3px', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "Variables used in the model are as below : ", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= '', style= {'width': '100%', 'background-color': '#D8E5F2', 
                                                            'border': 'none', 'height': '1px'}),
                    dbc.ListGroupItem(children= "• PPU : Derived directly from Nielsen data", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• TDP : Derived directly from Nielsen data", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• PPU Ratio : Custom Calculation as shown below", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• TDP Ratio : Custom Calculation as shown below", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• SOG(Source of Growth) SOM", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '50px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• Definitions", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• TDP Ratio Definitions : For a given manufacturer (X) selected in Page 1", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "PepsiCo TDP Ratio = PepsiCo TDP / X's TDP ", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 'text-indent': '50px',
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• PPU Ratio Definitions : For a given manufacturer (X) selected in Page 1", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "PepsiCo PPU Ratio = PepsiCo PPU / X's PPU ", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 'text-indent': '50px',
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "• SOG (Source of Growth) SOM", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 
                                              'font-size': '85%', 'height': '33px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(children= "PepsiCO SOG in Channel 'X' = PepsiCO Sales in Channel 'X' /  PepsiCO sales in all channels", 
                                      style= {'width': '100%', 'color': 'black',
                                              'background-color': '#D8E5F2', 'border': 'none', 'text-indent': '50px',
                                              'font-size': '85%', 'height': '50px', 'font-weight': 'bold', 'font-family': 'Verdana'}),
                    
                ])
                
            ], style= {'border': 'none'}
                
            ),
        ], width= 12),
                
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ]),
    
    
    
], fluid = True, style = {'background-color': '#F9F9F9'})