# 第一个完整项目：用 train\.py 训练，用 test\.py 测试 CIFAR\-10 模型

前面的 `Transform`、`DataLoader`、网络、损失函数和优化器，最终都在两个脚本里连起来：`train.py` 用 CIFAR\-10 训练、评估和保存模型；`test.py` 用自己的单张图片做推理。本文只记录当前真实运行的代码，不把尚未完成的功能写成已经实现。



## `model.py`：把网络结构单独放进一个文件



将 `Huai` 网络写入 `model.py`，可以让 `train.py` 只负责训练，`test.py` 只负责加载模型和预测。三个文件的分工会更清楚：



```Plain Text
model.py  -> 定义 Huai 网络结构
train.py  -> 读取数据、训练、评估、保存模型
test.py   -> 读取图片、加载模型、单张图片预测
```



```Python
import torch
from torch import nn

class Huai(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    huaixu = Huai()
    input_data = torch.ones((64, 3, 32, 32))
    output = huaixu(input_data)
    print(output.shape)  # torch.Size([64, 10])
```



网络接收 `[64, 3, 32, 32]` 的一批 CIFAR\-10 图片，最后输出 `[64, 10]` 的 logits。`if __name__ == "__main__":` 中的代码只会在直接运行 `python model.py` 时执行；当 `train.py` 使用 `from model import Huai` 导入网络时，不会自动跑这段检查代码。



注意构造函数和入口判断分别应写为 `__init__`、`__name__`、`__main__`，即前后各两个下划线；聊天中出现的 `**init**` 或 `**name**` 是 Markdown 显示造成的，不能直接复制进 Python 文件。



## `train.py`：准备 CIFAR\-10 数据



### 1\. 导入模块并读取数据集



`model.py` 已经定义了 `Huai` 网络，所以训练脚本只需要导入它。训练集与测试集都使用 `ToTensor()`，将图片转换成模型能够接收的张量。



```Python
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
```



`root="./data"` 指定数据保存位置，`train=True` 读取训练集，`train=False` 读取测试集；`download=True` 表示本地没有数据时自动下载。



### 2\. 查看数据量，并按 batch 加载



先打印数据集长度，确认数据确实读取成功。再用 `DataLoader` 每次取 64 张图片：



```Python
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
```



一个 batch 的图片 shape 是 `[64, 3, 32, 32]`，标签 shape 是 `[64]`。当前代码没有设置 `shuffle=True`，因此训练集按固定顺序读取；这是之后可以尝试优化的一项。



## `train.py`：配置网络、loss 与优化器



这段代码创建模型、分类损失函数和 SGD 优化器。`total_train_step`、`total_test_step` 分别记录训练和测试到了第几个 batch，便于作为 TensorBoard 的横坐标。



```Python
huaixu = Huai()

loss_function = nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.SGD(huaixu.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 3

writer = SummaryWriter("logs_train")
```



`CrossEntropyLoss` 直接接收网络输出的 logits 和类别编号标签，因此网络最后一层不需要手动添加 `Softmax`。



## 使用 GPU 训练：两种写法



GPU 训练的核心规则只有一条：**参与同一次运算的模型、输入图片和标签，必须在同一设备上。** 否则会出现 CPU Tensor 和 CUDA Tensor 不能一起计算的报错。



### 写法一：逐项使用 `.cuda()`



第一种写法在检测到 CUDA 可用时，分别把模型、损失函数和每个 batch 的数据移动到 GPU：



```Python
huaixu = Huai()
loss_function = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    huaixu = huaixu.cuda()
    loss_function = loss_function.cuda()

for imgs, targets in train_dataloader:
    if torch.cuda.is_available():
        imgs = imgs.cuda()
        targets = targets.cuda()

    outputs = huaixu(imgs)
    loss = loss_function(outputs, targets)
```



这种方式直观，也能在没有 GPU 的电脑上自动留在 CPU。但在训练和测试循环里都要重复写 `if torch.cuda.is_available()`，代码会逐渐变长。



### 写法二：统一使用 `device` 和 `.to(device)`



第二种写法先把设备保存到一个变量里，再用同一套 `.to(device)` 调用移动模型和数据。这是更推荐的写法：既简洁，又能同时兼容 GPU 和 CPU。



```Python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

huaixu = Huai().to(device)
loss_function = nn.CrossEntropyLoss().to(device)

for imgs, targets in train_dataloader:
    imgs = imgs.to(device)
    targets = targets.to(device)

    outputs = huaixu(imgs)
    loss = loss_function(outputs, targets)
```



源文件第二份代码使用 `device = torch.device("cuda")`，这会强制要求电脑存在可用 GPU；若在 CPU 环境运行会报错。上面的条件写法更稳妥。`CrossEntropyLoss` 本身没有可学习参数，移动它通常不是关键步骤，但统一写成 `.to(device)` 没问题，也便于以后替换为带状态的模块。



测试阶段同样要移动数据：



```Python
huaixu.eval()
with torch.no_grad():
    for imgs, targets in test_dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = huaixu(imgs)
```



GPU 只改变计算发生的位置，不改变训练步骤本身：`zero_grad()`、`backward()`、`step()` 的顺序完全不变。



## `train.py`：每一轮训练做了什么



外层循环控制训练 3 轮，内层循环从 `train_dataloader` 依次取出 batch。一个 batch 内依次完成：前向传播、计算 loss、清空旧梯度、反向传播、优化器更新参数。每训练 100 个 batch，把 loss 写入 TensorBoard。



```Python
for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i + 1))

    for data in train_dataloader:
        imgs, targets = data

        # 1. 前向传播：得到 [batch_size, 10] 的类别得分
        outputs = huaixu(imgs)

        # 2. 计算这一批预测与真实标签之间的误差
        loss = loss_function(outputs, targets)

        # 3. 计算梯度，并让优化器更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. 记录训练次数和训练 loss
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
```



`optimizer.zero_grad()` 必须在 `loss.backward()` 前面，因为 PyTorch 默认会累积梯度。`loss.item()` 将单个 loss 张量转成普通数值，适合打印和写入 TensorBoard。



## `train.py`：每轮训练后测试模型



训练完一轮后，代码遍历整个测试集。测试阶段不调用反向传播，也不更新参数；`torch.no_grad()` 会关闭梯度计算，减少不必要的内存和计算开销。



```Python
total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data

            # 1. 得到这一批图片的 10 类得分
            outputs = huaixu(imgs)

            # 2. 累加每个 batch 的 loss
            loss = loss_function(outputs, targets)
            total_test_loss += loss.item()
            total_test_step += 1

            # 3. 取每张图得分最高的类别，并累计正确图片数
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

            writer.add_scalar("test_loss", total_test_loss, total_test_step)
            if total_test_step % 20 == 0:
                print("测试次数：{}，Loss：{}".format(total_test_step, loss.item()))

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
```



`outputs.argmax(1)` 表示沿类别这一维寻找最大得分的位置，得到每张图片的预测类别编号。`total_accuracy` 是正确图片的总数，再除以 `test_data_size`，就是这一轮在整个测试集上的准确率。



当前日志中的 `test_loss` 是测试过程中从第一个 batch 开始累计的 loss；`test_accuracy` 则在每轮测试结束后记录一次整体结果。



## `train.py`：保存模型并关闭日志



当前代码在每轮测试结束后保存完整模型，因此训练 3 轮会得到 `huaixu_0.pth`、`huaixu_1.pth` 和 `huaixu_2.pth`。最后关闭 `SummaryWriter`，确保日志写入完成。



```Python
torch.save(huaixu, "huaixu_{}.pth".format(i))
    print("模型已保存")

writer.close()
```



因为这里保存的是完整模型对象，之后通过 `torch.load()` 加载时，环境中必须能找到 `Huai` 类的定义。对于来源不明的模型文件，不应设置 `weights_only=False` 去加载；这里仅加载自己训练生成或确认可信的文件。



## `test.py`：准备自己的图片



测试集准确率回答的是“模型整体表现怎样”；单张图片推理则回答“模型会把这张图片预测成什么”。测试图片必须被处理成与训练输入一致的三通道、`32 × 32` Tensor。



```Python
import torch
import torchvision
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = "./imgs/dog3.png"
image = Image.open(image_path)
print(image)

# PNG 可能带透明通道；转换后统一为 RGB 三通道
image = image.convert("RGB")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
])

image = transform(image)
print(image.shape)  # torch.Size([3, 32, 32])
```



`convert("RGB")` 能让 PNG 的透明通道不影响模型输入，也兼容原本就是 RGB 的 JPG 图片。当前训练只使用了 `ToTensor()`，所以推理只额外加入了尺寸调整；如果以后训练加入归一化，推理必须同步加入相同的归一化。



准备的图片如下，可直接截图保存：



![airplane1\.png](图片和附件/airplane1.png)

![dog\.png](图片和附件/dog.png)

![dog2\.png](图片和附件/dog2.png)

![dog3\.png](图片和附件/dog3.png)



## `test.py`：定义模型、加载模型并预测



模型一次接收一个 batch，因此单张图片要增加 batch 维度，得到 `[1, 3, 32, 32]`。模型和图片还必须在同一设备上。下面代码中的 `Huai` 类与 `model.py` 的定义相同；它必须在 `torch.load()` 前已经定义或导入。



```Python
# Huai 类必须已经定义，或通过 from model import Huai 导入
model = torch.load(
    "./huaixu_gpu_29.pth",
    map_location=device,
    weights_only=False,
)
print(model)

# 将一张图片补成一个 batch，并移动到与模型相同的设备
image = torch.reshape(image, (1, 3, 32, 32))
image = image.to(device)

# eval() 切换为推理模式；no_grad() 关闭梯度计算
model.eval()
with torch.no_grad():
    output = model(image)

print(output)           # shape 为 [1, 10] 的 logits
print(output.argmax(1)) # 得分最高的类别编号
```



`output` 的 shape 为 `[1, 10]`：1 代表一张输入图片，10 代表 CIFAR\-10 的 10 个类别得分。`argmax(1)` 取这 10 个得分中最大的那个位置，这就是模型预测的类别编号。



当前测试脚本读取的是 `huaixu_gpu_29.pth`，来自`train_gpu_1.py`；运行时要确认加载的文件名与希望测试的模型一致。



序号对应的类别名称

![5bb8ce48eda5cbbff51133631c665c75\.png](图片和附件/5bb8ce48eda5cbbff51133631c665c75.png)



若要将类别编号显示成名称，可以补上这段代码：

```Python
classes = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
prediction = output.argmax(1).item()
print("预测类别：", classes[prediction])
```



## 对当前 train\.py 与 test\.py 的补充说明



这两份脚本已经构成了完整的“训练—评估—保存—单图预测”闭环。为了让代码的行为和博客描述完全一致，还需要知道以下几点：



- **显式导入 ****`torch`**：`train.py` 使用了 `torch.optim.SGD` 和 `torch.save`，建议在文件开头加上 `import torch`，不要依赖 `from model import *` 间接带入它。

- **训练集是否打乱**：当前 `train_dataloader = DataLoader(train_data, batch_size=64)` 会使用默认的 `shuffle=False`。正式训练时通常改为 `shuffle=True`；测试集保持 `False` 即可。

- **训练与测试模式**：当前网络没有 Dropout 或 BatchNorm，所以没有写 `huaixu.train()` / `huaixu.eval()` 也能运行。但推荐在每轮训练开始前调用 `huaixu.train()`，测试循环开始前调用 `huaixu.eval()`，以后换网络时不会遗漏。

- **测试 loss 的含义**：你现在记录的 `total_test_loss` 是每个 batch 的 loss 累加值，因而随测试 batch 数递增。若想比较每轮平均 loss，可记录 `total_test_loss / len(test_dataloader)`；若想看每个 batch 的变化，则写入单次的 `loss.item()`。

- **模型文件名**：训练脚本生成 `huaixu_0.pth`、`huaixu_1.pth`、`huaixu_2.pth`，而测试脚本加载 `huaixu_gpu_29.pth`。运行前要确认它们是否属于同一模型版本。

- **完整模型加载**：训练时使用 `torch.save(huaixu, ...)` 保存完整模型，因此测试前必须定义或导入 `Huai` 类。`weights_only=False` 只应用于自己保存或确认可信的模型文件。

    

可以将训练和测试两个关键位置写成：

```Python
# 每轮训练开始前
huaixu.train()

# 测试开始前
huaixu.eval()
with torch.no_grad():
    # 遍历 test_dataloader，计算 loss 和正确数
    pass
```



## 当前代码的完整闭环

```Plain Text
CIFAR-10 数据集
  -> DataLoader 每批 64 张图
  -> Huai 网络输出 10 类 logits
  -> CrossEntropyLoss 计算误差
  -> SGD 反向传播更新参数
  -> 测试集统计 loss 和准确率，写入 TensorBoard
  -> 保存 huaixu_{轮数}.pth
  -> test.py 读取模型，对 dog3.png 进行单张预测
```



