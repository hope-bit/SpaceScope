from galaxy_mnist import GalaxyMNIST

# 64 pixel images
dataset = GalaxyMNIST(
    root='/some/download/folder',
    download=True,
    train=True  # by default, or set False for test set
)

images, labels = dataset.data, dataset.targets

(custom_train_images, custom_train_labels), (custom_test_images, custom_test_labels) = dataset.load_custom_data(test_size=0.8, stratify=True)
