import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)


class TestLossNetwork(nn.Module):
    def __init__(self):
        super(TestLossNetwork, self).__init__()
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
tln = TestLossNetwork()
for data in dataloader:
    imgs, targets = data
    outputs = tln(imgs)
    print(outputs)
    print(targets)
    result_loss = loss(outputs, targets)
    result_loss.backward()  # 反向传播