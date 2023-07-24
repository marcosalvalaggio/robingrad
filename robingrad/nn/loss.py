from robingrad.tensor import Tensor
from typing import List

class MSELoss:
    def __init__(self, output: List[Tensor], target: List[Tensor]):
        self.output = output
        self.target = target

    def __call__(self) -> "Tensor":
        return ((self.output - self.target)**2).mean()
        