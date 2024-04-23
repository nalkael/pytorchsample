import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url

print(torch.__version__)

dataset_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
download_url(dataset_url, './data/cifar10')

with tarfile.open('./data/cifar10/cifar-10-python.tar.gz', 'r:gz') as tar:
    tar.extractall(path='./data/cifar10')