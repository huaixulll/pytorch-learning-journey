import ssl

import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

ssl._create_default_https_context = ssl._create_unverified_context

dataset = torchvision.datasets.CIFAR10("./data", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataLoader = DataLoader(dataset, batch_size=64, drop_last=True)  # drop_last True 最后一页不足舍弃


class Huai(nn.Module):
    def __init__(self):
        super(Huai, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


huaixu = Huai()
for data in dataLoader:
    imgs, target = data
    print(imgs.shape)

    # output = torch.reshape(imgs, (1, 1, 1, -1))  #将图像展开成一行
    # print(output.shape)

    output = torch.flatten(imgs)  # 将[64, 3, 32, 32] -> 转换为 [1, 1, 1, 196608]
    print(output.shape)

    output = huaixu(output)
    print(output.shape)
