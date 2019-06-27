import torch
import torch.nn as nn

# n w h c
test_data = torch.randn(size=[1, 28, 28, 3])
# print(test_data.size())
# n c w h
test_data = test_data.transpose(3, 1)

# print(test_data.size())

conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
out = conv1(test_data)
print("conv1:", out.size())

pool2 = nn.MaxPool2d(kernel_size=2)
out = pool2(out)
print("pool2: ", out.size())

conv3 = nn.Conv2d(in_channels=16, out_channels=400, kernel_size=5)
out = conv3(out)
print("conv3: ", out.size())

conv4 = nn.Conv2d(in_channels=400, out_channels=400, kernel_size=1)
out = conv4(out)
print("conv4: ", out.size())

conv5 = nn.Conv2d(in_channels=400, out_channels=4, kernel_size=1)
out = conv5(out)
print("conv5: ", out.size())
softmax = nn.Softmax(dim=1)
z = softmax(out)
print("softmax: ", z.size())
# pool4 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
# out = pool4(out)
# print(out.size())


#
# m = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1)
# input = torch.randn(1, 192, 28, 28)
# output = m(input)
#
# print(output.size())

