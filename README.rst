.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/SpaceScope.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/SpaceScope
    .. image:: https://readthedocs.org/projects/SpaceScope/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://SpaceScope.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/SpaceScope/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/SpaceScope
    .. image:: https://img.shields.io/pypi/v/SpaceScope.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/SpaceScope/
    .. image:: https://img.shields.io/conda/vn/conda-forge/SpaceScope.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/SpaceScope
    .. image:: https://pepy.tech/badge/SpaceScope/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/SpaceScope
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/SpaceScope

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==========
SpaceScope
==========


Download the hdf5 dataset locally, and then update the root path.


A longer description of your project goes here...
## Dataset Selection
The dataset used in this project is the GalaxyMNIST dataset, consisting of galaxy images and their corresponding labels. The labels represent different types of galaxies. The dataset is stored in an HDF5 file format and contains both image data (in the form of pixel values) and label data.

## Types of Plots Chosen
- **Histogram of Label Distribution**: This histogram shows the distribution of the galaxy labels across the dataset, allowing us to see how many instances exist for each galaxy type.
- **Grid of Sample Images**: Randomly chosen images from the dataset, each paired with its corresponding label. This provides a visual representation of the dataset.

## Instructions for Running the Code
1. **Dependencies**: Install the required libraries using:
    ```bash
    pip install matplotlib numpy dash pillow h5py
    ```
2. **Static Plots**: Run the functions `plot_label_histogram` and `plot_image_grid` to display the histograms and image grid.
3. **Interactive Plot**: To run the Dash app, use the command:
    ```bash
    python app.py
    ```
    The app will allow you to select labels and view the corresponding images interactively.

## Interactive Visualization Guide
- Use the dropdown to select one or more labels.
- The images corresponding to the selected labels will be displayed below the dropdown.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
