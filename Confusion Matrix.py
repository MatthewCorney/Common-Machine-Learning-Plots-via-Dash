import plotly.figure_factory as ff
import plotly.io as pio
from sklearn.metrics import confusion_matrix
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash

# Takes y_true and y_pred as inputs which then forms a confusion matrix with options to normalize or keep the raw counts
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

# Build App
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

markdown = (
    '''#### Confusion matrix
    
Confusion matrix of the predicted class against the true class

''')

# Layout of the page
app.layout = html.Div([
    dcc.Markdown(markdown),
    html.Label(['Normalize?', dcc.RadioItems(id='selection-radioitem',
                                             value='True', options=[{'label': 'True', 'value': 'True'},
                                                                    {'label': 'False', 'value': 'False'}])]),
    dcc.Graph(id='graph')])


# Define callback to update graph
@app.callback(
    Output('graph', 'figure'),
    [Input("selection-radioitem", "value")]
)
def plot_confusion_matrix(normalize):
    if normalize == 'True':
        cf = confusion_matrix(y_true, y_pred, normalize='all')
        title = 'Normalized confusion matrix'
    else:
        cf = confusion_matrix(y_true, y_pred, normalize=None)
        title = 'Raw Confusion Matrix'
    z = np.flip(np.flip(cf, 1))
    z_text = z.round(2)
    fig = ff.create_annotated_heatmap(z=z, annotation_text=z_text, x=list(range(0, cf.shape[0])),
                                      y=list(range(0, cf.shape[1])), colorscale='gray')
    fig.update_layout(xaxis=dict(title='Predicted Class', side="bottom"),
                      yaxis=dict(title='True Class'),
                      title=title,
                      template='seaborn')

    return fig


app.run_server(mode='External')
