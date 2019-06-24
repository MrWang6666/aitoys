import torch
import torch.nn as nn
from torch.utils.data import dataloader
from torchvision import transforms, datasets

MNIST_PATY = r"/Users/magichao/PycharmProjects/helloai/MNIST/MNIST_data"
EPOCH = 100
BATCH_SIZE = 100

# 数据归一化对象
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 数据集加载
train_dataset = datasets.MNIST(MNIST_PATY, train=True, transform=data_transform, download=False)
train_dataload = dataloader.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.MNIST(MNIST_PATY, train=True, transform=data_transform, download=False)
test_dataload = dataloader.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)



# 定义网络模型
class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self

    def forward(self, *input):
        pass

    def __len__(self):
        pass

