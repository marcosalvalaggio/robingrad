from robingrad import Tensor
import torch
import unittest
from micrograd.engine import Value
import numpy as np

class TestTensor(unittest.TestCase):

    def test_tensor_ops(self):
        # robin
        a = Tensor.eye(3, requires_grad=True)
        b = Tensor.full((3,3), 3., requires_grad=True)
        c = Tensor.ones((3,3), requires_grad=True)
        loss = ((a+b-c)**2).sum()
        loss.backward()
        res_robin = a.grad.tolist()
        # torch
        a = torch.eye(3, requires_grad=True)
        b = torch.full((3,3),3.,requires_grad=True)
        c = torch.ones((3,3), requires_grad=True)
        loss = ((a+b-c)**2).sum()
        loss.backward()
        res_torch = a.grad.numpy().tolist()
        # test 
        self.assertAlmostEqual(res_robin, res_torch)

    def test_robin_micrograd(self):
        # robin
        a = Tensor(-4.0, requires_grad=True)
        b = Tensor(2.0, requires_grad=True)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f
        #print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
        g.backward()
        res_robin = np.round(float(a.grad),4)
        #print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
        #print(f'{b.grad:.4f}')
        # micrograd
        a = Value(-4.0)
        b = Value(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f
        #print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
        g.backward()
        #print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
        #print(f'{b.grad:.4f}')
        res_micro = np.round(float(a.grad),4)
        # test 
        self.assertAlmostEqual(res_robin, res_micro)

    def test_matmul(self):
        # robin 
        a = Tensor.ones((2,3), requires_grad=True)
        b = Tensor.full((3,2), 3., requires_grad=True)
        c = a @ b
        loss = c.sum()
        loss.backward()
        res_robin_1 = a.grad.tolist()
        res_robin_2 = b.grad.tolist()
        # torch 
        a = torch.ones((2,3), requires_grad=True)
        b = torch.full((3,2), 3., requires_grad=True)
        c = a @ b
        loss = c.sum()
        loss.backward()
        res_torch_1 = a.grad.numpy().tolist()
        res_torch_2 = b.grad.numpy().tolist()
        # test 
        self.assertEqual(res_robin_1, res_torch_1)
        self.assertEqual(res_robin_2, res_torch_2)





