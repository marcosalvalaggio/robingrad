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

    def test_reshape_matmul(self):
        # robin 
        a = Tensor.ones((3,2), requires_grad=True)
        aa = a.reshape((2,3))
        b = Tensor.full((3,2), 3., requires_grad=True)
        c = aa @ b
        loss = c.sum()
        loss.backward()
        res_robin_1 = a.grad.tolist()
        # torch 
        a = torch.ones((3,2), requires_grad=True)
        aa = a.reshape((2,3))
        b = torch.full((3,2), 3., requires_grad=True)
        c = aa @ b
        loss = c.sum()
        loss.backward()
        res_torch_1 = a.grad.numpy().tolist()
        # test
        self.assertEqual(res_robin_1, res_torch_1)

    def test_slice(self):
        #robin
        a = Tensor.ones((3,3), requires_grad=True)
        aa = a[0:2]
        b = Tensor.full((3,2), 3., requires_grad=True)
        c = aa @ b
        loss = c.sum()
        loss.backward()
        res_robin_1 = a.grad.tolist()
        res_robin_2 = b.grad.tolist()
        # torch
        a = torch.ones((3,3), requires_grad=True)
        aa = a[0:2]
        b = torch.full((3,2), 3., requires_grad=True)
        c = aa @ b
        loss = c.sum()
        loss.backward()
        res_torch_1 = a.grad.numpy().tolist()
        res_torch_2 = b.grad.numpy().tolist()
        #test
        self.assertEqual(res_robin_1, res_torch_1)
        self.assertEqual(res_robin_2, res_torch_2)

    def test_transpose(self):
        #robin
        a = Tensor.ones((3,2), requires_grad=True)
        b = a.T
        c = Tensor.full((3,2), 3., requires_grad=True)
        d = b @ c
        loss = d.sum()
        loss.backward()
        res_robin_1 = a.grad.tolist()
        res_robin_2 = c.grad.tolist()
        # torch
        a = torch.ones((3,2), requires_grad=True)
        b = torch.transpose(a,1,0)
        c = torch.full((3,2), 3., requires_grad=True)
        d = b @ c
        loss = d.sum()
        loss.backward()
        res_torch_1 = a.grad.numpy().tolist()
        res_torch_2 = c.grad.numpy().tolist()
        #test
        self.assertEqual(res_robin_1, res_torch_1)
        self.assertEqual(res_robin_2, res_torch_2)

    def test_mean(self):
        # robin
        a = Tensor.eye(3, requires_grad=True)
        b = Tensor.full((3,3), 3., requires_grad=True)
        c = a @ b
        d = c.mean(axis=0, keepdim=True)
        e = Tensor.full((1,3), 4., requires_grad=True)
        f = d * e
        loss = f.sum()
        loss.backward()
        res_robin_1 = a.grad.tolist()
        # torch 
        a = torch.eye(3, requires_grad=True)
        b = torch.full((3,3), 3., requires_grad=True)
        c = a @ b
        d = c.mean(axis=0, keepdim=True)
        e = torch.full((1,3), 4., requires_grad=True)
        f = d * e
        loss = f.sum()
        loss.backward()
        res_torch_1 = a.grad.numpy().tolist()
        # test 
        self.assertAlmostEqual(res_robin_1, res_torch_1)

    def test_new_add(self):
        # robin 1
        a = Tensor.full((5,1), 3., requires_grad=True)
        b = Tensor.full((1,1), 2., requires_grad=True)
        c = a + b
        loss = c.mean()
        loss.backward()
        res_robin_1 = a.grad.tolist()
        res_robin_2 = b.grad.tolist()
        # torch 1
        a = torch.full((5,1), 3., requires_grad=True)
        b = torch.full((1,1), 2., requires_grad=True)
        c = a + b
        loss = c.mean()
        loss.backward()
        res_torch_1 = a.grad.tolist()
        res_torch_2 = b.grad.tolist()
        #test 1
        self.assertEqual(res_robin_1, res_torch_1)
        self.assertEqual(res_robin_2, res_torch_2)
        # robin 2
        a = Tensor.full((5,1), 3., requires_grad=True)
        b = Tensor.full((1,1), 2., requires_grad=True)
        c =  b + a
        loss = c.mean()
        loss.backward()
        res_robin_3 = a.grad.tolist()
        res_robin_4 = b.grad.tolist()
        # torch 2
        a = torch.full((5,1), 3., requires_grad=True)
        b = torch.full((1,1), 2., requires_grad=True)
        c = b + a
        loss = c.mean()
        loss.backward()
        res_torch_3 = a.grad.tolist()
        res_torch_4 = b.grad.tolist()
        #test 2
        self.assertEqual(res_robin_3, res_torch_3)
        self.assertEqual(res_robin_4, res_torch_4)

    def test_flatten(self):
        # robin 
        a = Tensor.ones((2,3), requires_grad=True)
        b = Tensor.full((3,2), 3., requires_grad=True)
        c = a @ b
        d = c.flatten()
        loss = d.mean()
        loss.backward()
        res_robin_1 = a.grad.tolist()
        res_robin_2 = b.grad.tolist()
        # torch
        a = torch.ones((2,3), requires_grad=True)
        b = torch.full((3,2), 3., requires_grad=True)
        c = a @ b
        d = c.flatten()
        loss = d.mean()
        loss.backward()
        res_torch_1 = a.grad.numpy().tolist()
        res_torch_2 = b.grad.numpy().tolist()
        # test 
        self.assertEqual(res_robin_1, res_torch_1)
        self.assertEqual(res_robin_2, res_torch_2)

    def test_new_mul(self):
        # robin 1
        a = Tensor.full((5,1), 3., requires_grad=True)
        b = Tensor.ones((1,1), requires_grad=True)
        c = a * b
        loss = c.sum()
        loss.backward()
        res_robin_1 = a.grad.tolist()
        res_robin_2 = b.grad.tolist()
        # torch 1
        a = torch.full((5,1), 3., requires_grad=True)
        b = torch.ones((1,1), requires_grad=True)
        c = a * b
        loss = c.sum()
        loss.backward()
        res_torch_1 = a.grad.numpy().tolist()
        res_torch_2 = b.grad.numpy().tolist()
        # test 1
        self.assertEqual(res_robin_1, res_torch_1)
        self.assertEqual(res_robin_2, res_torch_2)
        # robin 2
        a = Tensor.full((5,1), 3., requires_grad=True)
        b = Tensor.ones((1,1), requires_grad=True)
        c = b * a
        loss = c.sum()
        loss.backward()
        res_robin_3 = a.grad.tolist()
        res_robin_4 = b.grad.tolist()
        # torch 2
        a = torch.full((5,1), 3., requires_grad=True)
        b = torch.ones((1,1), requires_grad=True)
        c = b * a
        loss = c.sum()
        loss.backward()
        res_torch_3 = a.grad.numpy().tolist()
        res_torch_4 = b.grad.numpy().tolist()
        # test 2
        self.assertEqual(res_robin_3, res_torch_3)
        self.assertEqual(res_robin_4, res_torch_4)

    def test_broadcasting(self):
        a = Tensor.ones((2,2))
        b = Tensor.zeros((1,))
        c = Tensor.broadcast(a, b)
        target_dim = (2,2)
        self.assertEqual(c.shape, target_dim)
        d = Tensor.ones((2,2))
        e = 1 - a
        self.assertEqual(e.shape, target_dim)


