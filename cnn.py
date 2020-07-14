"""Convolutional Neural Network."""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

import torchvision
from torchvision.datasets import MNIST


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

def main():
    # 首先载入数据，这里直接使用PyTorch的MNIST数据集
    train_data = MNIST(root='./data', train=True, download=True, transform=None)
    test_data = MNIST(root='./data', train=False, download=True, transform=None)
    #print(train_data)
    #print(test_data)

    


if __name__ == '__main__':
    main()