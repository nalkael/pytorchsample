import os
import torch
import torchvision
import torchaudio
import tarfile
from torchvision.datasets.utils import download_url

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
