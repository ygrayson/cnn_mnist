"""Convolutional Neural Network."""


import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST


class CNN(nn.Module):
    def __init__(self):
        """init."""
        super(CNN, self).__init__()

        # 卷积网络层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 完全连接网络层
        self.fc1 = nn.Linear(in_features=7*7*10, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
    
    def forward(self, x):
        """forward."""
        # 第一层卷积和取最大值
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        # 第二层卷积和取最大值
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        # 完全连接层
        x = x.view(-1, 7*7*10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x



def visualize_data(train_data, test_data):
    """visualize data."""
    # 数据集的基本信息
    print(train_data)
    print(test_data)
    print("Size of training data:", train_data.data.shape)

    # 展示前40张手写数字
    num_of_images = 40
    for index in range(1, num_of_images + 1):
        plt.subplot(4, 10, index)
        plt.axis('off')
        plt.imshow(train_data.data[index], cmap='gray_r')
    plt.show()
    
def main():
    # 首先载入数据，这里直接使用PyTorch的MNIST数据集
    print("===============Loading MNIST Dataset================")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_data)

    # 看看数据是啥样儿
    print("===============Visualizing Data==============")
    #visualize_data(train_data, test_data)

    # 定义神经网络和训练参数
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)
    batch_size = 100
    epoch_num = int(train_data.data.shape[0]) // batch_size

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    # 训练神经网络
    print("===============Training CNN==================")
    print("Total Training Epoch: {}".format(epoch_num))
    for epoch in range(1, epoch_num+1):
        # 每个batch一起训练，更新神经网络weights
        for idx, (img, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print("Training Epoch {} Completed".format(epoch))
        if epoch == 2:
            break
        

    # 在测试集上测试
    print("================Testing the CNN==================")
    total = 0
    correct = 0
    for i, (test_img, test_label) in enumerate(test_loader):
        # 正向通过神经网络得到预测结果
        outputs = model(test_img)
        predicted = torch.max(outputs.data, 1)[1]
        print("Correct label is", test_label)
        print("Prediction is", predicted)
        
        # 总数和正确数
        total += len(test_label)
        correct += (predicted == test_label).sum()
    
    accuracy = correct / total
    print('Testing Results:\n  Loss: {}  \nAccuracy: {} %'.format(loss.data, accuracy*100))


if __name__ == '__main__':
    main()