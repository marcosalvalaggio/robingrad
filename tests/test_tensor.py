from robingrad import Tensor
import torch
import unittest

class TestTensor(unittest.TestCase):
    def test_tensor_ops(self):
        # robin
        a = Tensor.eye(3)
        b = Tensor.full((3,3), 3)
        c = Tensor.ones((3,3))
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





