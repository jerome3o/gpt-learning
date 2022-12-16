import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root="fashion_data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root="fashion_data",
    train=False,
    download=True,
    transform=ToTensor(),
)


