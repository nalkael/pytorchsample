import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import tarfile
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from PIL import Image

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

    # Show file information
    def show_info(self, file_path = ''):
        if os.path.exists(file_path):
            #get file state information
            file_stat = os.stat(file_path)
            # file size
            file_size = file_stat.st_size

            # print file information
            print(f"File: {file_path}")
            print(f"Size: {file_size}")
    
    # Load image dataset for Directory
    def load_image_dataset(self, dataset_dir='', transform=ToTensor()):
        if os.path.exists(dataset_dir):
            pass
        else:
            print(f'Directory of \"{dataset_dir}\" does not exist.')