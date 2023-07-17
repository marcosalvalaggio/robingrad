from tinygrad.state import get_parameters
import numpy as np
from typing import TypeVar

ModelType = TypeVar('ModelType')

class Optimizer:
    def __init__(self, model: ModelType):
        self.params = get_parameters(model)

    def zero_grad(self):
        for param in self.params: param.grad = np.zeros_like(param.grad)
        
