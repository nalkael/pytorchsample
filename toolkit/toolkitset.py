import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
#from torch.utils.data import random_split
import tarfile

class Toolkits:
    def __init__(self):
       pass 

    # Download the dataset
    def download_dataset(self, dataset_url="", file_dir=""):
        download_url(dataset_url, file_dir)

    
    # Extract the packed file
    def extract_file(self, file_path = "", type = 'r:gz', extract_dir = ""):
        with tarfile.open(file_path, type) as tar:
            tar.extractall(extract_dir)
            tar.close()