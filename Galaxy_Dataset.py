import os
import panel as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import param
from tqdm import tqdm
from galaxy_mnist import GalaxyMNIST
from io import BytesIO
import base64

class GalaxyDashboard(param.Parameterized):
    """
    A dashboard for visualizing GalaxyMNIST images.

    Attributes:
        class_filter (param.ObjectSelector): Filter for galaxy classes.
    """
    class_filter = param.ObjectSelector(default='All', objects=['All'])
    
    def __init__(self, **params):
        """
        Initializes the dashboard with GalaxyMNIST dataset.

        Args:
            **params: Additional parameters for the dashboard.
        """
        super().__init__(**params)
        self.progress = pn.widgets.Progress(name='Initializing Dashboard', value=0, max=100, sizing_mode='stretch_width')
        self.image_data = self.load_images_with_metadata()
        self.filtered_data = self.image_data.copy()
        self.data_table = pn.widgets.Tabulator(self.filtered_data, name='Image Data', sizing_mode='stretch_width', height=400)
        self.entry_counter = pn.pane.Markdown(f"<div style='font-size: 32px; font-weight: bold;'>Total Displayed Entries: {len(self.filtered_data)}</div>")
        self.preview_button = pn.widgets.Button(name='Preview Images', button_type='primary')
        self.image_gallery = pn.Row()
        self.class_plot_pane = pn.pane.Matplotlib()
        self.brightness_histogram_pane = pn.pane.Matplotlib()
        self.scatter_plot_pane = pn.pane.Matplotlib()
        self.param.class_filter.objects = ['All'] + sorted(self.image_data['class'].unique().tolist())
        self.create_static_graphs()
        self.update_dashboard()

        # Add a callback to update the image gallery when the button is pressed
        self.preview_button.on_click(self.update_image_gallery)

    def load_images_with_metadata(self):
        """
        Loads images and metadata from the GalaxyMNIST dataset.

        Returns:
            pd.DataFrame: A DataFrame containing the image metadata.
        """
        dataset = GalaxyMNIST(root='C:/Users/HopeW/Downloads/MEM679_Project/SpaceScope/galaxy_mnist', download=True, train=True)
        image_data = []
        total_files = len(dataset)

        for idx, (image, label) in enumerate(tqdm(dataset, desc="Loading Galaxy Images")):
            image_array = np.array(image)
            average_brightness = np.mean(image_array)
            
            metadata = {
                'index': idx,
                'class': label,
                'average_brightness': average_brightness,
                'width': image.size[0],
                'height': image.size[1],
                'image': image
            }
            image_data.append(metadata)
            self.progress.value = int((idx + 1) / total_files * 100)  # Update progress bar

        df = pd.DataFrame(image_data)  # Convert the list to a pandas DataFrame
        return df

    def create_static_graphs(self):
        """
        Creates static graphs for the dashboard.
        """
        # Plot for distribution of galaxy types
        class_counts = self.image_data['class'].value_counts()
        fig, ax = plt.subplots()
        class_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Galaxy Type Distribution')
        ax.set_xlabel('Galaxy Type')
        ax.set_ylabel('Count')
        ax.set_xticklabels(class_counts.index, rotation=0)  # Set text horizontal
        ax.grid(True, axis='y')  # Add horizontal gridlines only
        self.class_plot_pane.object = fig  # Update the plot

        # Histogram of brightness
        fig2, ax2 = plt.subplots()
        ax2.hist(self.image_data['average_brightness'], bins=20, color='purple')
        ax2.set_title('Image Brightness Distribution')
        ax2.set_xlabel('Average Brightness')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, axis='y')  # Add horizontal gridlines only
        self.brightness_histogram_pane.object = fig2  # Update the plot

        # Scatter plot for brightness vs. size
        fig3, ax3 = plt.subplots()
        ax3.scatter(self.image_data['average_brightness'], self.image_data['width'] * self.image_data['height'], color='green')
        ax3.set_title('Scatter Plot of Brightness and Image Size')
        ax3.set_xlabel('Average Brightness')
        ax3.set_ylabel('Image Area (Width x Height)')
        ax3.grid(True, axis='both')  # Add gridlines
        self.scatter_plot_pane.object = fig3  # Update the plot

    @param.depends('class_filter', watch=True)
    def update_dashboard(self):
        """
        Updates the dashboard based on the selected filters.
        """
        self.filtered_data = self.image_data.copy()
        if self.class_filter != 'All':
            self.filtered_data = self.filtered_data[self.filtered_data['class'] == self.class_filter]
        self.data_table.value = self.filtered_data
        self.entry_counter.object = f"<div style='font-size: 32px; font-weight: bold;'>Total Entries Displayed: {len(self.filtered_data)}</div>"

    def update_image_gallery(self, event):
        """
        Updates the image gallery based on the filtered data.

        Args:
            event: The event that triggered the update.
        """
        self.image_gallery.clear()  # Clear the current gallery
        images = []
        for _, row in self.filtered_data.iterrows():
            images.append(pn.Column(
                pn.pane.PNG(row['image'], width=100, height=100),
                pn.pane.Markdown(f"<div style='text-align: center;'>{row['class']}</div>", width=100)
            ))
        self.image_gallery.extend(images)

    def view(self):
        """
        Creates the layout for the dashboard.

        Returns:
            pn.Column: The layout of the dashboard.
        """
        header = pn.pane.HTML(
            """
            <div style='background-color: maroon; color: white; padding: 10px; text-align: center; width: 100%;'>
                <h1>GalaxyMNIST Visualization Dashboard</h1>
                <p>This dashboard allows you to filter and visualize galaxy images from the GalaxyMNIST dataset.</p>
            </div>
            """,
            sizing_mode='stretch_width'
        )

        return pn.Column(
            header,
            self.progress,
            pn.Row(
                self.class_plot_pane,
                pn.pane.Markdown("**Galaxy Type Distribution**: This plot shows the frequency of each galaxy type in the dataset.", width=300)
            ),
            pn.Row(
                self.brightness_histogram_pane,
                pn.pane.Markdown("**Image Brightness Distribution**: This histogram shows the average brightness levels of the galaxy images.", width=300)
            ),
            pn.Row(
                self.scatter_plot_pane,
                pn.pane.Markdown("**Brightness vs. Image Area**: This scatter plot displays the relationship between brightness and area of galaxy images.", width=300)
            ),
            pn.Row(
                pn.Column(
                    pn.Param(self.param, widgets={'class_filter': pn.widgets.Select}),
                    self.entry_counter,
                    self.preview_button,
                )
            ),
            pn.Column(
                pn.Row(self.image_gallery, sizing_mode='stretch_width'),
                sizing_mode='stretch_width', scroll=True, height=400
            ),
            self.data_table
        )

if __name__ == "__main__":
    pn.extension()
    dashboard = GalaxyDashboard()
    pn.serve(dashboard.view)
