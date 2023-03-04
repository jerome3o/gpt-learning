import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# Datasets handle storage (and extraction?)
# DataLoaders wrap Datasets in an iterable for use in code

# Loading in pre-made fashion dataset
training_data = datasets.FashionMNIST(
    root="fashion_data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="fashion_data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Iterating and visualising the dataset

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# Moved into a notebook for visualisation

