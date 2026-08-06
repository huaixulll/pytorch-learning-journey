# 数据如何进入模型：Transforms、Dataset 与 DataLoader

模型不认识 JPEG、PNG 或文件夹，它只接收 Tensor。于是，数据管道（从原始图片到一个 batch 的完整过程）最重要的不是“下载了哪个数据集”，而是弄清楚每一层分别负责什么：`Transform` 改变样本，`Dataset` 提供单个样本，`DataLoader` 组织一批样本。



## 从一张图片到一个 Tensor



在 CIFAR\-10 中，一张彩色图像经过 `ToTensor()` 后，会从图像对象变成形如 `[3, 32, 32]` 的张量，像素值通常被缩放到 `[0, 1]`。多个操作应该交给 `Compose` 按顺序执行：



```Python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])
```



当前 `train.py` 只使用了 `ToTensor()`，还没有做归一化，因此单张图片推理也只需要使用 `Resize + ToTensor()`。以后若加入归一化，推理脚本必须使用同一组归一化参数。



## `Dataset`：按索引取一条样本



`torchvision.datasets.CIFAR10` 已经封装好了样本读取。它会在被索引时返回 `(image, target)`，即一张处理后的图片和对应标签。



```Python
from torchvision import datasets

train_set = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform,
    download=True,
)

image, target = train_set[0]
print(image.shape, train_set.classes[target])
```



## `DataLoader`：把样本组成 batch



训练不会每次只送一张图片。`DataLoader` 将多个样本打包，给模型一个 batch：



```Python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_set,
    batch_size=64,
)

images, targets = next(iter(train_loader))
print(images.shape)   # torch.Size([64, 3, 32, 32])
print(targets.shape)  # torch.Size([64])
```



这里有一个好记的分工：Dataset 决定“第 *i* 条样本是什么”，DataLoader 决定“这次给模型哪一批样本”。当前 `train.py` 没有显式传入 `shuffle`，因此使用默认值 `False`，即按固定顺序读取训练集。后续可以尝试为训练集加入 `shuffle=True`；测试集仍保持默认值即可。



注释：

`shuffle` 表示是否在每一轮开始前，把数据顺序随机打乱。

- `shuffle=True`：每轮图片顺序都不同，通常用于训练集，能减少模型记住固定顺序的风险。

- `shuffle=False`：保持数据原始顺序，通常用于测试集，方便稳定评估和排查问题。



## 本篇小任务



打印一个 batch 的 `images.shape` 和 `targets.shape`，再把第一张图及其类别名写入 TensorBoard。只要这一步正确，后面的模型就能接上稳定的输入。

