import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2,2,3))
        self.bias = nn.Parameter(torch.ones(2,3))
        # self.init_weights(nlhb=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = x @ self.weight
        print(x.shape)
        print(x)
        x += self.bias.unsqueeze(1)
        print(self.bias.unsqueeze(1).shape)
        print(x.shape)
        return x

a = torch.ones((2, 2, 2))

li = A()

print(li(a))