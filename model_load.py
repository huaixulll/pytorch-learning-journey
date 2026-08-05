import torch
import torchvision
from torch import nn

# 或者解决方法  将model_save中的代码引入
from model_save import *

# 方式一保存后的模型加载
model = torch.load("vgg16_method1.pth")
# print(model)

# 方式二加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
model = vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(vgg16)
# 输出为字典
# print(model)

# 陷阱1，
# 解决
# class Test1(nn.Module):
#     def __init__(self):
#         super(Test1, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, input):
#         output = self.conv1(input)
#         return output


model = torch.load("test_method1")  # 报错AttributeError: Can't get attribute 'Test1' on <module '__main__'
print(model)