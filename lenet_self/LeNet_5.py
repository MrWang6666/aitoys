import torch
import torch.nn as nn
from torch.utils.data import dataloader
from torchvision import transforms, datasets
from torch import optim


MNIST_PATH = r"/Users/magichao/PycharmProjects/helloai/MNIST/MNIST_data"
EPOCH = 1
BATCH_SIZE = 100

# 数据归一化对象
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 数据集加载
train_dataset = datasets.MNIST(MNIST_PATH, train=True, transform=data_transform, download=False)
train_dataload = dataloader.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.MNIST(MNIST_PATH, train=True, transform=data_transform, download=False)
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
            nn.ReLU(inplace=False),
            nn.MaxPool2d()
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

        loss = loss_fn(out, lable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('epoch: {},i: {}, loss: {:.3}'.format(epoch, i, loss.data.to(device).item()))  # 损失值显示3位

# 评估模型
net.eval()
# 评估损失累积
eval_loss = 0
# 累加精度
eval_acc = 0
for i, (img, lable) in enumerate(test_dataload):

    out = net(img)

    loss = loss_fn(out, lable)
    # 求输出的最大值索引
    pred_out = torch.argmax(out, dim=1)

    num_correct = (pred_out == lable).sum()
    eval_acc += num_correct.item()
    eval_loss += loss.data.to(device).item() * lable.size(0)



print("out data: \n", torch.argmax(out, 1))
print("lable data: \n", lable)
print(torch.max(out, 1))

print('Test Loss: {:.3}, Acc: {:.3}'.format(
    # 测试数据集里的平均损失，累计损失➗测试集中样本数量
    eval_loss / (len(test_dataset)),
    # 同上
    eval_acc / (len(test_dataset))
))


