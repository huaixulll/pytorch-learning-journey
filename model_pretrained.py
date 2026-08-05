import torchvision
from torch import nn

# 报错，The dataset is no longer publicly accessible 不能公开下载，只能下载后替换跟目录文件
# 此文件train 训练集图片大小为147.9GB
# torchvision.datasets.ImageNet('./dataset', split='train', download=True,
#                               transform=torchvision.transforms.ToTensor())

# 数据量为528M
# 网络模型参数是未经过训练的,初始化的参数
vgg16_false = torchvision.models.vgg16(pretrained=False)
# 网络模型参数是经过训练的
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

# 向vgg16中添加模块 使得由1000类输出为10类
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
# 指定步骤由4096类输出10类
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)