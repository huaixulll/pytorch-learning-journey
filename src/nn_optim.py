import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset)


class TestOptim(nn.Module):
    def __init__(self):
        super(TestOptim, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        result = self.model1(input)
        return result


loss = nn.CrossEntropyLoss()
to = TestOptim()
# torch.optim.SGD 算法  参数、学习速率
optim = torch.optim.SGD(to.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = to(imgs)
        result_loss = loss(outputs, targets)

        # 将网路参数中每个梯度调节为0
        optim.zero_grad()
        # 调用反向传播
        result_loss.backward()
        # 对每个参数进行调优
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)
