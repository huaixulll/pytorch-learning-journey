# 入门学习 PyTorch：跑通一条完整训练链路

第一次学 PyTorch 时，我记下了不少零散的模块：`DataLoader` 怎么写、`nn.Conv2d` 有哪些参数、损失函数怎么调用、模型又该怎样保存。回头再看，这些笔记并没有错，但它们更像一盒还没拼起来的零件。



这次我想换一种方式：不再按 API 名称记忆，而是围绕一个问题推进——**怎样让一个模型从读入图片开始，完成训练、评估，并留下可复现实验记录？**



以 CIFAR\-10 图像分类为例，一次完整训练大致是这样的：

```Plain Text
原始图片
  -> Transform 预处理
  -> Dataset 按索引提供样本
  -> DataLoader 组成 batch
  -> 模型前向传播，输出 logits
  -> 损失函数衡量预测与标签的差距
  -> 反向传播计算梯度
  -> 优化器更新参数
  -> TensorBoard 记录曲线，测试集评估并保存模型
```

这条链路也是本系列的目录。之后每一篇只多解决一个问题，最后再把它们合起来。我的目标不是马上追求很高的准确率，而是先建立一个判断框架：数据是什么形状、模型输出代表什么、loss 在何时下降、参数又是在何时被更新。



CIFAR\-10的说明：

https://docs\.pytorch\.org/vision/stable/generated/torchvision\.datasets\.CIFAR10\.html\#torchvision\.datasets\.CIFAR10

CIFAR\-10的下载地址（下载Python版本）：

https://cave\.cs\.toronto\.edu/kriz/cifar\.html



## 学习前的三个约定

第一，代码能运行不等于理解。每次调用一个模块时，我都会先写下它的输入、输出和 shape。例如，CIFAR\-10 的一个 batch 经过 `ToTensor()` 后常见形状为 `[64, 3, 32, 32]`：64 张图、3 个通道、高和宽均为 32。



第二，训练过程要可见。只看终端里最后一行准确率，常常无法判断问题来自数据、学习率还是模型结构。因此系列一开始就引入 TensorBoard，而不是把它当作最后才补的“可视化装饰”。



第三，先用小模型跑通。一个能稳定训练、能保存、能复现的小 CNN，比一段看不懂的大模型代码更适合作为起点。



## 本系列完成后的最小成果



到最后，我希望得到的不只是一个 `.pth` 文件，而是一套可复用的最小工程：

```Plain Text
project/
├── model.py          # 模型定义
├── train.py          # 训练、验证和日志
└── logs/             # TensorBoard 日志
```



## 系列预告



1. 先看见训练：用 TensorBoard 记录实验

2. 数据如何进入模型：Transforms、Dataset 与 DataLoader

3. 从层到网络：理解 `nn.Module`

4. 模型为什么会学习：损失函数、反向传播与优化器

5. 第一个完整项目：训练并评估 CIFAR\-10 CNN

    

下一篇从 TensorBoard 开始。先让 loss、accuracy 和数据样本看得见，后面的调试才有依据。

