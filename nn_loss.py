# 1、计算实际输出和目标之间的差距  2、为我们更新输出提供一定的依据（反向传播）

from torch import nn
import torch

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# loss 差
loss = nn.L1Loss(reduction='sum')
result = loss(inputs, targets)
print(result)

# 均方差
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)

# 交叉熵损失函数
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)