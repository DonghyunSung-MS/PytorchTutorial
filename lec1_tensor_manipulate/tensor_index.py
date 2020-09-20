import torch

batch_size = 10
features = 25
x = torch.rand(batch_size, features)
print(x[0, :])#batch 0 , all features

x = torch.arange(10)
print(torch.where(x>5, x, x*2)) # statement, if , else
