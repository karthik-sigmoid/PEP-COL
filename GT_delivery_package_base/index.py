# Import required libraries
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Importing from app.py file
from app import server
from app import app

# import all pages in the app
from pages import home, Simulation, ErrorMetrics, scenario

# NavBar at the top of each page
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/PEPSICO_logo.png", height="30px"), width = 2),
                        dbc.Col(dbc.NavbarBrand("PepsiCo Guatemala - SOM Prediction Dashboard v2", 
                                                style = {'color': '#01529C', 'font-family': 'Verdana',
                                                         'font-weight': 'bold', 
                                                        'marginLeft': '140px',
                                                        'font-size': '180%', 'align': 'center'}), width=10),
                    ],
                    align="center", justify = "center",
                    no_gutters=True
                ),
                # href="/Simulation",
                style={"textDecoration": "none"}
            ),
        ], fluid= True
    ),
    color="#E4EDED"
)

# embedding the navigation bar to the layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

# pathname --> page contents
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    """This function takes pathname as an input and returns the corresponding page layout. 
    Whenever we clicks on the tab at the top after navbar, this will navigate the user to that respective page

    Args:
        pathname (url): Url for respective pages

    Returns:
        layout: Page layout
    """
    if pathname == '/ErrorMetrics':
        return ErrorMetrics.layout
    elif pathname == '/home':
        return home.layout
    elif pathname == '/scenario':
        return scenario.layout
    else:
        return Simulation.layout

# To run the app
if __name__ == '__main__':
    app.run_server(port=80, host= '0.0.0.0', debug=False)