import os
import torch
import torchvision
import torchaudio
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid

# print(torch.__version__)
# print(torchvision.__version__)
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


dataset = toolkit.load_image_dataset('./data/cifar10/train', ToTensor())
"""
each element is a tuple
contains a image tensor and a label
each image tensor in the shape [3, 32, 32] (32 * 32 pixel with 3 channels - RGB)
"""
#image, label = dataset[0]
#print(image.shape)
#print(image)

#list of classes is stored in the .classes property
print(dataset.classes)
img, label = dataset[0]

plt.rcParams['font.size'] = 12

def show_example(img, label):
    print(f'Label: {dataset.classes[label]} ({label})')
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

"""
show_example(*dataset[0])
show_example(*dataset[30])
"""

""" 
training and dataset

Training set
Validation set
Test set
"""
random_seed = 77
torch.manual_seed(random_seed)
# print(len(dataset))

# a small portion(5000) from traning data as valiation data
val_size = 5000
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# print(len(train_dataset)), print(len(val_dataset))


# create data loaders for training and validation
# load data in batches
batch_size = 64

train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(train_dataset, batch_size*2, shuffle=True, num_workers=2, pin_memory=True)

"""
show batch images from the dataset
with matplot library
more details to look up into
"""
# look at the batches of images from the dataset using the make_grid method from torchvision
def show_batch(dataload):
    dataload_iter = iter(dataload)
    images, labels = next(dataload_iter)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks([]); ax.set_yticks([])
    ax. imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    plt.show()

show_batch(train_dl)
show_batch(val_dl)

"""
define a simple convolutional neural network
"""

# a foo kernel function
def apply_kernel(image, kernel):
    ri, ci = image.shape
    rk, ck = kernel.shape
    ro, co = ri - rk + 1, ci - ck + 1
    output = torch.zeros([ro, co])


