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
test_data = datasets.FashionMNIST(
    root="fashion_data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using {device} device")

