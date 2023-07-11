from robingrad import Tensor
import torch

a = Tensor.eye(3)
b = Tensor.full((3,3), 3)
c = Tensor.ones((3,3))
loss = ((a+b-c)**2).sum()
print(loss)