import os
import torch
import torchvision
import torchaudio
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split

#print(torch.__version__)
from toolkit.toolkitset import Toolkits

toolkit = Toolkits()

toolkit.download_dataset(dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz', 
                         file_dir = './data')

toolkit.extract_file(file_path = './data/cifar10.tgz', 
                     type = 'r:gz', extract_dir = './data')

toolkit.show_info(file_path = './data/cifar10.tgz')

dir_list = os.listdir(path = './data/cifar10')
print(dir_list)

classes = os.listdir(path= './data/cifar10/test')
print(classes)

# example of cifar10 dataset
data_dir = './data/cifar10'
airplane_files = os.listdir('./data/cifar10/test/airplane')
print(f'No of test examples of airplanes: {len(airplane_files)}')
print(airplane_files[5:10])


dataset = Toolkits.load_image_dataset('./data/some')
