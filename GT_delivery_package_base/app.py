# Importing Dash libraries
import dash
import dash_bootstrap_components as dbc
import dash_auth

# Styleshets to incorporate bootstrap theme
# added a line to check git
external_stylesheets = [dbc.themes.BOOTSTRAP, 
                        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css', 
                        'https://use.fontawesome.com/releases/v5.8.1/css/all.css']

# initializing the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# To successfully deploy the app on server
server = app.server
app.config.suppress_callback_exceptions = True

# auth = dash_auth.BasicAuth(
#     app,
#     {'SigmoidDashboard': 'Sigmoid@123'}
# )