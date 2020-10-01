from sklearn.linear_model import LinearRegression
from sklearn import datasets
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import numpy as np
import plotly.io as pio

# Takes pd df as an imput to form the plots

# Load Data
df = datasets.load_iris(as_frame=True)['data']

# Build App
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

markdown = (
    '''#### Data Relationship Plots
    
Plots to show the relationship between two features in a dataset                 
''')

# Layout of the page
app.layout = html.Div([
    dcc.Markdown(markdown),
    dcc.Graph(id='graph'),
    html.Label(["Feature 1",
                dcc.Dropdown(id='feature1-dropdown',
                             clearable=False, value=df.columns[0],
                             options=[{'label': c, 'value': c} for c in set(df.columns)]
                             )]),
    html.Label(["Feature 2",
                dcc.Dropdown(id='feature2-dropdown',
                             clearable=False, value=df.columns[1],
                             options=[{'label': c, 'value': c} for c in set(df.columns)]
                             )])
])


# Define callback to update graph
@app.callback(
    Output('graph', 'figure'),
    [Input("feature1-dropdown", "value"),
     Input("feature2-dropdown", "value")]
)
def plot_data_relationship(var1, var2):
    if var1 != var2:
        fig = go.FigureWidget()
        # Get linear regression
        reg = LinearRegression().fit(np.array(df[var1]).reshape(-1, 1),
                                     np.array(df[var2]).reshape(-1, 1))
        linear_regression = reg.predict(np.array(df[var1]).reshape(-1, 1))

        fig.add_trace(go.Scatter(x=df[var1], y=df[var2], mode='markers', showlegend=False, ))

        fig.add_trace(go.Histogram(x=df[var1], name='x density', marker=dict(color='#1f77b4'), showlegend=False,
                                   yaxis='y2'))
        fig.add_trace(go.Histogram(y=df[var2], name='y density', marker=dict(color='#1f77b4'), showlegend=False,
                                   xaxis='x2'))
        fig.add_trace(go.Scatter(name='Regression', x=df[var1], y=linear_regression.flatten(), mode='lines',
                                 line={'color': 'black', 'dash': 'dash'}))

        fig.layout = dict(
            xaxis=dict(domain=[0, 0.77], showgrid=True, gridcolor='#EAEAEA', linecolor='black', zeroline=True,
                       showticklabels=True, title=var1),
            yaxis=dict(domain=[0, 0.77], showgrid=True, gridcolor='#EAEAEA', linecolor='black', zeroline=True,
                       showticklabels=True, title=var2),
            bargap=0.01,
            xaxis2=dict(domain=[0.78, 1], showgrid=False, zeroline=True),
            yaxis2=dict(domain=[0.78, 1], showgrid=False, zeroline=True),
            template='seaborn',
            plot_bgcolor='#FFFFFF',
            autosize=True,
            title=(var1 + ' against ' + var2))
    else:
        fig = ff.create_distplot([df[var1]], group_labels=[var1], colors=['#1f77b4'], show_rug=False)
        fig.layout = dict(xaxis=dict(title=var1),
                          yaxis=dict(title='Proportion'),
                          template='seaborn',
                          plot_bgcolor='#FFFFFF',
                          autosize=True,
                          showlegend=False,
                          title=var1 + ' Distribution')
    return fig


app.run_server(mode='External')
