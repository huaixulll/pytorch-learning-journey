# 先看见训练：用 TensorBoard 记录实验

训练模型时最容易出现一种错觉：程序没有报错，于是我以为它在正常学习。实际上，loss 可能根本没有下降，标签可能错位，或者模型只是在记住训练集。TensorBoard 的价值就在这里——它把训练过程中原本不可见的变化，变成可检查的曲线、图片和结构图。



## `SummaryWriter` 是什么



`SummaryWriter` 可以理解为一个记录器：训练脚本负责把数据写入日志目录，TensorBoard 负责把日志显示在浏览器中。它们的分工很简单：PyTorch 训练模型，TensorBoard 观察训练过程。



当前的 `train.py` 使用 `logs_train` 作为日志目录，并记录训练 loss、累计测试 loss 和整体测试准确率：



```Python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs_train")

# 训练阶段记录
total_train_step = 0
for data in train_dataloader: 
total_train_step = total_train_step + 1
# 每训练 100 个 batch 记录一次
if total_train_step % 100 == 0:
    writer.add_scalar("train_loss", loss.item(), total_train_step)

# 测试阶段记录
writer.add_scalar("test_loss", total_test_loss, total_test_step)
writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
```



运行训练脚本后，在终端输入下面命令，启动 TensorBoard：



```Bash
tensorboard --logdir=logs_train
```



打开终端提示的本地地址，就能看到曲线。当前日志名称为 `train_loss`、`test_loss`、`test_accuracy`；后续有多个实验时，也可以改成 `train/loss`、`test/loss` 这样的分层名称。



## 练手实例：记录一张图片和函数曲线



在正式训练模型之前，我先用一张蜜蜂图片和函数 `y = 2x` 熟悉 TensorBoard 的两种基本记录方式：`add_image()` 记录图片，`add_scalar()` 记录数值随步数的变化。



```Python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# 1. 创建日志目录
writer = SummaryWriter("logs")

# 2. 读取图片，并转为 NumPy 数组
image_path = "dataset/val/bees/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg"
img_pil = Image.open(image_path)
img_array = np.array(img_pil)

print(type(img_array))   # numpy.ndarray
print(img_array.shape)   # 例如 (高度, 宽度, 通道数)

# 3. 将图片写入 TensorBoard
writer.add_image("test", img_array, 2, dataformats="HWC")

# 4. 记录 y = 2x 的曲线
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

# 5. 关闭记录器，确保内容写入日志
writer.close()
```



这段代码可分为五步：创建记录器、读取图片、将图片转换为数组、写入图片和标量数据、关闭记录器。代码中的 `2` 是图片记录的步数，当前案例只写一张图片，因此它主要用于熟悉 `add_image()` 的参数。



这里 `img_array` 的图片数据排列是 **HWC**：高度（Height）、宽度（Width）、颜色通道（Channel）。而 PyTorch 的图片张量经常使用 **CHW** 排列，所以在 `add_image()` 中明确写出 `dataformats="HWC"` 很重要；否则 TensorBoard 可能把维度理解错，导致图片显示异常。



运行后，Python结果为：

![image\.png](图片和附件/image%201.png)

在当前目录执行：

```Bash
tensorboard --logdir=logs
```



打开终端显示的本地地址，可以在 Images 页面看到蜜蜂图片，在 Scalars 页面看到一条从 `(0, 0)` 延伸到 `(99, 198)` 的直线。结果

![image\.png](图片和附件/image.png)

![image\.png](图片和附件/image%202.png)

## 我最常记录的三类信息



**标量** 用于 loss、accuracy 和学习率。它们告诉我“训练有没有在往正确方向走”。



```Python
writer.add_scalar("train/loss", loss.item(), global_step)
writer.add_scalar("val/accuracy", accuracy, epoch)
```



**图像** 用于检查预处理后的样本。当前训练脚本还没有写入图片；之后在做裁剪、归一化或数据增强时，可以补上这一步，先看一眼输入通常比盯着报错信息更快。



```Python
writer.add_image("samples/image", image, global_step)
```



**网络图** 用于确认模型的大致连接关系。当前脚本还没有写入网络图；它适合作为后续辅助检查，但不必为了“画图”牺牲代码可读性。



```Python
writer.add_graph(model, example_batch)
```



## 读曲线时需关注的三个问题



1. 训练 loss 是否总体下降？如果完全不动，优先检查学习率（lr=learning\_rate）、标签和 loss 的输入。

2. 训练集指标和验证集指标是否逐渐分开？训练集越来越好、验证集反而停滞，通常意味着开始过拟合。

3. 曲线是否剧烈震荡？先检查学习率是否过大，再看 batch size 和数据是否存在异常。

    

## 本篇小任务



运行 `tensorboard --logdir=logs_train`，确认当前的 `train_loss`、`test_loss` 和 `test_accuracy` 都已出现。之后再补充图像记录，检查输入图片是否正确。

