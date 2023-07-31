from robingrad.tensor import Tensor
from typing import List, Union

class MSELoss:
    def __call__(self, output: Union[List[Tensor], Tensor], target: Union[List[Tensor], Tensor]) -> Tensor:
        return ((output - target)**2).mean()
        

class BCELoss:
    def __call__(self, output: Union[List[Tensor], Tensor], target: Union[List[Tensor], Tensor]) -> Tensor:
        neg_output = 1-output
        return (-(target * output.log() + (1-target)*neg_output.log())).mean()