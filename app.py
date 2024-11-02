import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from data_preparation import load_data

# Load the data
data = load_data()

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='feature-selector',
        options=[{'label': col, 'value': col} for col in data.columns],
        value=data.columns[0]
    ),
    dcc.Graph(id='scatter-plot')
])

@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('feature-selector', 'value')]
)
def update_figure(selected_feature):
    fig = px.scatter(data, x=selected_feature, y='another_feature', color='category_column')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
