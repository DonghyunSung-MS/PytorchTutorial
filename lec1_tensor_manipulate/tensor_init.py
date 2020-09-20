import torch

print(torch.__version__)

device = 'cuda' if torch.cuda.is_available() else "cpu"

ex_tensor = torch.tensor([[1,2,3],[4,5,6]], 
                        dtype=torch.float32,
                        device=device,
                        requires_grad=True)

x = torch.empty(size = (3, 3))
x = torch.zeros(3, 3)
x = torch.randn(3, 3)
x = torch.ones(3, 3)
x = torch.eye(3, 3)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
x = torch.empty(size=(1,5)).uniform_(0, 1)

tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())#int16
print(tensor.long())#int64
print(tensor.half())#float16
print(tensor.float())#float32
print(tensor.double())#float64

import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
back2np_array = tensor.numpy()