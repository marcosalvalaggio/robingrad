from tinygrad.tensor import Tensor
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
        self.first_moments = [np.zeros_like(param.data) for param in self.params]
        self.second_moments = [np.zeros_like(param.data) for param in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.params):
            self.first_moments[i] = self.beta1*self.first_moments[i] + (1-self.beta1)*param.grad
            self.second_moments[i] = self.beta2*self.second_moments[i] + (1-self.beta2)*(param.grad**2)
            first_moment_corrected = self.first_moments[i] / (1-self.beta1**self.t)
            second_moment_corrected = self.second_moments[i] / (1-self.beta2**self.t)
            param.data -= (self.lr * first_moment_corrected)/(np.sqrt(second_moment_corrected) + self.eps)
            
