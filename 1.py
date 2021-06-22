import torch
import torch.nn as nn

a = torch.ones((2, 2, 2))

li = nn.Linear(2, 3)


print(li(a[0]).shape)
print(a.shape)