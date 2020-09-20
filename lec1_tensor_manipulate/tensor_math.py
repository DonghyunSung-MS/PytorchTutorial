import torch

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)
z = x + y

z = x - y

z = x/y
print(z)

#inplace operation
t = torch.zeros(3)
t.add_(x)
t += x
print(t)

#exponentiatioz
z = x.pow(2)

#Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

t1 = torch.rand(batch, n, m)
t2 = torch.rand(batch, m, p)

t3 = torch.bmm(t1, t2)
print(t3.shape)

#*Broadcasting: elementarizing operation by expanding or copying to match bigger shape
x1 = torch.rand(5, 5)
x2 = torch.rand(1, 5)#(1, 5) ->(copy) (5, 5)

z = x1 - x2
z = x1 ** x2

#other operation
x = torch.arange(6)#0,1,2,3,4,5,
z = torch.clamp(x, min = 3, max = 4)#3,3,3,3,4,4
print(z)


