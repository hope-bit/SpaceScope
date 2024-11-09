# Import necessary libraries
import dash
from dash import dcc, html
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import dash.dependencies
import plotly.graph_objects as go

# Load the GalaxyMNIST dataset
# Assuming data is in the form of NumPy arrays, e.g., X_train.npy and y_train.npy
X_train = np.load(r'C:/Users/HopeW/Downloads/MEM679_Project/SpaceScope/X_train.npy')  # Images
y_train = np.load(r'C:/Users/HopeW/Downloads/MEM679_Project/SpaceScope/y_train.npy')  # Labels

# Check the shape of the loaded data
print("Shape of X_train:", X_train.shape)  # Should be (num_samples, img_height, img_width)
print("Shape of y_train:", y_train.shape)  # Should be (num_samples,)

# Initialize the Dash app
app = dash.Dash(__name__)

# Prepare the figure for the histogram (distribution of labels)
label_hist = px.histogram(y_train, title="Distribution of Galaxy Labels", labels={'value': 'Galaxy Class'})

# Prepare the figure for a sample image display
def create_image_figure(image_array, title="Sample Image"):
    fig = go.Figure()
    img = Image.fromarray(image_array)
    fig.add_trace(go.Image(z=np.array(img)))
    fig.update_layout(title=title, xaxis={'showticklabels': False}, yaxis={'showticklabels': False})
    return fig

# Default image (first sample)
sample_img_figure = create_image_figure(X_train[0], title="Sample Galaxy Image")

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children="GalaxyMNIST Interactive Dashboard"),
    
    html.Div(children="""
        This is a simple Dash app to visualize the GalaxyMNIST dataset.
        Use the controls below to interact with the data.
    """),
    
    # Dropdown for selecting plot type
    dcc.Dropdown(
        id='plot-type-dropdown',
        options=[
            {'label': 'Label Distribution Histogram', 'value': 'label_hist'},
            {'label': 'Sample Image', 'value': 'sample_img'}
        ],
        value='label_hist',  # Default value is histogram
        style={'width': '50%'}
    ),
    
    # Graph for displaying the selected plot
    dcc.Graph(id='plot-output', figure=label_hist)
])

# Update the figure based on the selected plot type
@app.callback(
    dash.dependencies.Output('plot-output', 'figure'),
    [dash.dependencies.Input('plot-type-dropdown', 'value')]
)
def update_plot(selected_plot):
    if selected_plot == 'sample_img':
        # Return a figure with a random sample image
        random_idx = np.random.randint(0, len(X_train))
        return create_image_figure(X_train[random_idx], title=f"Sample Image {random_idx}")
    return label_hist  # Default to the label distribution histogram

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
