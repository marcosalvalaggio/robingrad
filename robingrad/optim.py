from tinygrad.state import get_parameters
import numpy as np
from typing import TypeVar

ModelType = TypeVar('ModelType')

class Optimizer:
    def __init__(self, model: ModelType):
        self.params = get_parameters(model)

    def zero_grad(self) -> None:
        for param in self.params: param.grad = np.zeros_like(param.grad)
        

class SGD(Optimizer):
    def __init__(self, model: ModelType, lr: float = 3e-4):
        super().__init__(model=model)
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            param.data += -self.lr * param.grad


# https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
class Adam(Optimizer):
    def __init__(self, model: ModelType, lr: float = 0.0001, beta1: float = 0.9, beta2: float =  0.999, eps: float = 1e-08):
        super().__init__(model=model)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps

    def step(self) -> None:
        for param in self.params:
            pass
