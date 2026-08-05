import torch
from torch import nn

class Huai(nn.Module):
    def __init__(self) :
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

huaixu = Huai()
x = torch.tensor(1.0)
output = huaixu(x)
print(output)