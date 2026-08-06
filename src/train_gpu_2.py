# 准备数据集
import torch
import torchvision
from torch import nn, device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from model import *

# 定义训练设备
device = torch.device("cuda")

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10，训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 神经网络搭建在 model.py 文件中
# 搭建神经网络
class Huai(nn.Module):
    def __init__(self):
        super(Huai, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
huaixu = Huai()
huaixu = huaixu.to(device)  # 使用GPU

# 创建损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)   # 使用GPU

# 优化器
learning_rate = 1e-2  # 0.01
optimizer = torch.optim.SGD(huaixu.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)    # 使用GPU
        targets = targets.to(device)    # 使用GPU
        outputs = huaixu(imgs)
        loss = loss_function(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 优化前要梯度清0
        loss.backward()  # 调用反向传播，同时计算梯度
        optimizer.step()  # 对参数优化

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0  # 整体测试loss
    total_accuracy = 0   # 整体正确个数
    with torch.no_grad():  # 测试阶段不更新参数，它会关闭梯度计算，节省内存并加快测试速度
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)  # 使用GPU
            targets = targets.to(device)  # 使用GPU
            outputs = huaixu(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            total_test_step = total_test_step + 1
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()  # argmax(1)为横向查找最大数的位置索引，0为纵向
            total_accuracy = total_accuracy + accuracy

            writer.add_scalar("test_loss", total_test_loss, total_test_step)
            if total_test_step % 20 == 0:
                print("测试次数：{}，Loss：{}".format(total_test_step, loss.item()))

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)

    torch.save(huaixu, "huaixu_{}.pth".format(i))
    print("模型已保存")

writer.close()

