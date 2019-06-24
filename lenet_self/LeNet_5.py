import torch
import torch.nn as nn
from torch.utils.data import dataloader
from torchvision import transforms, datasets
from torch import optim

MNIST_PATY = r"/Users/magichao/PycharmProjects/helloai/MNIST/MNIST_data"
EPOCH = 100
BATCH_SIZE = 30

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
        # n c h w
        # input: 1, 28, 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=False)
            # nn.MaxPool1d()
        )# batch, 8, 12, 12 | (28 - 5 + 2*0) / 2 + 1


        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=False)
        )# batch, 16, 6, 6 | (12 - 3 + 2*0) / 2 + 1

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=False)
        )# batch, 32, 3, 3 | (6 - 3 + 2*1) / 2 + 1

        self.fc = nn.Sequential(
            nn.Linear(in_features=32 * 3 * 3, out_features=10),
            nn.BatchNorm1d(num_features=10),
            nn.Softmax()
        )
    def forward(self, input):
        y1 = self.conv1(input)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)

        y3 = y3.reshape(y3.size(0), -1)

        out = self.fc(y3)

        return out


net = LeNet_5()

# 定义loss，使用交叉熵loss
loss_fn = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(params=net.parameters())

# 开始训练
net.train()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

for epoch in range(EPOCH):
    for i, (img, lable) in enumerate(train_dataload):

        img = img.to(device)

        out = net(img)

        print(out.size())

        loss = loss_fn(out, lable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('epoch: {},i: {}, loss: {:.3}'.format(epoch, i, loss.data.to(device).item()))  # 损失值显示3位

