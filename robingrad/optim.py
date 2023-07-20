from robingrad.tensor import Tensor
import numpy as np
from typing import List

class Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = params

    def zero_grad(self) -> None:
        for param in self.params: 
            param.grad = np.zeros_like(param.grad)
        

class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 3e-4):
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            param.data += -self.lr * param.grad


# https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float =  0.999, eps: float = 1e-08):
        super().__init__(params)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(param.data) for param in self.params] # first moment
        self.v = [np.zeros_like(param.data) for param in self.params] # second moment 
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*param.grad
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*(param.grad**2)
            m_hat = self.m[i] / (1-self.beta1**self.t) # first moment corrected
            v_hat = self.v[i] / (1-self.beta2**self.t) # second moment corrected
            param.data -= (self.lr * m_hat)/(np.sqrt(v_hat) + self.eps)
            
