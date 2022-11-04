import torch
import numpy as np
from misc import *
from hsic import *
from model import *
torch.cuda.is_available()

import torch.nn.functional as F
import torchvision.transforms as transforms


from torchvision.datasets import CIFAR10, MNIST


def load_mnist(dataFolderPath='./data/mnist', train=True, download=True, batchSize=64):
    
    train_transform = transforms.Compose([transforms.ToTensor()]) # , transforms.Resize(size=(227, 227))
    valid_transform = train_transform

    train_set = MNIST(dataFolderPath, train=train,
                  download=download, transform=train_transform)
    valid_set = MNIST(dataFolderPath, train=False,
                  download=True, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=batchSize, shuffle=False)    

    return train_loader, val_loader


def load_cifar10(dataFolderPath='./data/cifar10', train=True, download=True, batchSize=64):
    
    train_transform = transforms.Compose([transforms.ToTensor()]) # , transforms.Resize(size=(227, 227))
    valid_transform = train_transform

    train_set = CIFAR10(dataFolderPath, train=train,
                  download=download, transform=train_transform)
    valid_set = CIFAR10(dataFolderPath, train=False,
                  download=True, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=batchSize, shuffle=False)    

    return train_loader, val_loader

def get_data(data_name, batch_size):

    if data_name=='cifar10':
        dataPath = './data/cifar10'
        train_loader, test_loader=load_cifar10(dataPath, batchSize=batch_size)

    elif data_name=='mnist':
        dataPath = './data/mnist'
        train_loader, test_loader=load_mnist(dataPath, batchSize=batch_size)

    return train_loader, test_loader