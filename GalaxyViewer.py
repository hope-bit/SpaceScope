import h5py
import matplotlib.pyplot as plt
import numpy as np
from dash import Dash, dcc, html, Input, Output
import panel as pn

# Load dataset from local
file_path = 'C:/Users/HopeW/Downloads/MEM679_Project/SpaceScope/galaxy_mnist/GalaxyMNIST/raw/train_dataset.hdf5'  # Adjust if needed
with h5py.File(file_path, 'r') as hdf5_file:
    images = hdf5_file['images'][:]
    labels = hdf5_file['labels'][:]

# Part 1: Static Visualizations using Matplotlib
def plot_label_histogram(labels):
    """
    Plot a histogram of label distribution.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(labels, bins=np.arange(labels.min(), labels.max() + 1), color='skyblue', edgecolor='black')
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_image_grid(images, labels, num_samples=10):
    """
    Display a grid of sample images with labels.
    """
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Call the static visualization functions
plot_label_histogram(labels)
plot_image_grid(images, labels)

# Part 2: Interactive Visualization using Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Image Label Explorer"),
    html.Label("Select Label:"),
    dcc.Dropdown(id='label-dropdown', options=[{'label': str(int(label)), 'value': int(label)} for label in np.unique(labels)], multi=True),
    html.Div(id='images-output')
])

@app.callback(
    Output('images-output', 'children'),
    [Input('label-dropdown', 'value')]
)
def update_images(selected_labels):
    if not selected_labels:
        return html.Div("Select a label to view images.")
    
    # Filter images and labels by selected label(s)
    selected_indices = np.isin(labels, selected_labels)
    filtered_images = images[selected_indices][:10]  # Limit display to 10 images for brevity

    # Display filtered images
    image_elements = []
    for i, img in enumerate(filtered_images):
        image_elements.append(html.Div([
            html.Img(src=f'data:image/png;base64,{np_to_base64(img)}', style={'height': '100px'}),
            html.P(f"Label: {labels[selected_indices][i]}")
        ], style={'display': 'inline-block', 'margin': '10px'}))

    return html.Div(image_elements)

# Helper function to convert numpy array to base64 string
import base64
from io import BytesIO
def np_to_base64(img_array):
    """
    Convert an image array to a base64-encoded string.
    """
    pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Run Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

# Part 3: Interactive Visualization using Panel
pn.extension()

def view_image(index=0, label=None):
    """
    Display a specific image with a dynamic selection slider.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(images[index])
    ax.set_title(f"Image Index: {index}, Label: {labels[index]}")
    plt.axis("off")
    return fig

# Panel layout
slider = pn.widgets.IntSlider(name='Image Index', start=0, end=len(images)-1, step=1)
label_selector = pn.widgets.Select(name='Label', options=list(np.unique(labels)))
panel = pn.Column("## Interactive Image Viewer", slider, label_selector, pn.bind(view_image, index=slider, label=label_selector))

# Display the panel
panel.show()

# Histogram of label distribution
def plot_label_histogram(labels):
    """
    Plot a histogram of label distribution.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(labels, bins=np.arange(labels.min(), labels.max() + 1), color='skyblue', edgecolor='black')
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Display sample images with labels
def plot_image_grid(images, labels, num_samples=10):
    """
    Display a grid of sample images with labels.
    """
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Call the functions to display the plots
plot_label_histogram(labels)
plot_image_grid(images, labels)
