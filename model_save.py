import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1，模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# 陷阱1
class Test1(nn.Module):
    def __init__(self):
        super(Test1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, input):
        output = self.conv1(input)
        return output


t1 = Test1()
torch.save(t1, "test_method1")