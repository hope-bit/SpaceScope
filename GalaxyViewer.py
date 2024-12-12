import h5py
import matplotlib.pyplot as plt
import numpy as np
from dash import Dash, dcc, html, Input, Output
import base64
from io import BytesIO
from PIL import Image
from galaxy_mnist import GalaxyMNISTHighrez

# Load dataset from local
file_path = 'C:/Users/HopeW/Downloads/MEM679_Project/SpaceScope/galaxy_mnist/GalaxyMNIST/train_dataset.hdf5'  # Update this path
with h5py.File(file_path, 'r') as hdf5_file:
    images = hdf5_file['images'][:]
    labels = hdf5_file['labels'][:]
    
classArray = ["smooth_round", "smooth_cigar", "edge_on_disk", "barred_spiral"]  # Class names for labels
# Replace the numeric labels with the corresponding class names
label_names = np.array([classArray[label] for label in labels])

# Function to convert image array to base64 string
def np_to_base64(img_array):
    pil_img = Image.fromarray((img_array * 255).astype(np.uint8))  # Normalize to 0-255 for image display
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Static Visualizations
# Static Visualizations - Bar Plot for Label Distribution
def plot_label_barplot(labels):
    """
    Plot a bar plot of label distribution.
    """
    # Count occurrences of each label
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(8, 6))
    plt.bar(unique_labels, label_counts, color='skyblue', edgecolor='black')
    plt.title('Label Distribution (Bar Plot)')
    plt.xlabel('Type of Galaxy')
    plt.ylabel('Number of Images per type')
    plt.xticks(unique_labels)  # Display label values on x-axis
    plt.grid(True)
    
    # Convert plot to base64 image
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.read()).decode('utf-8')


def plot_image_grid(images, labels, num_samples=10):
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {labels[i]}")  # Use label_names here
        plt.axis("off")
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.read()).decode('utf-8')

# Dash App Setup
app = Dash(__name__)

app.layout = html.Div([
    html.H1("GalaxyMNIST Image Label Explorer"),
    
    # Static Visualizations Section
    html.Div([
        html.H2("Label Distribution (Bar Plot)"),
        html.Img(src=f"data:image/png;base64,{plot_label_barplot(label_names)}", style={'width': '80%'}),
        
        html.H2("Sample Images Grid"),
        html.Img(src=f"data:image/png;base64,{plot_image_grid(images, label_names)}", style={'width': '80%'}),
    ], style={'margin-bottom': '40px'}),  # Margin between static and interactive sections
    
    # Interactive Visualization Section
    html.Div([
        html.H2("Interactive Visualization: Select Label"),
        dcc.Dropdown(id='label-dropdown', options=[{'label': str(int(label)), 'value': int(label)} for label in np.unique(labels)], multi=True),
        html.Div(id='images-output')
    ])
])

# Callback to update images based on dropdown selection
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
            html.P(f"Label: {label_names[selected_indices][i]}")  # Use label_names here
        ], style={'display': 'inline-block', 'margin': '10px'}))

    return html.Div(image_elements)

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
