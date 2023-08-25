from robingrad.tensor import Tensor
from .loss import MSELoss, BCELoss

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True, mean: float = 0., std: float = 1.):
        self.weight = Tensor.normal(mean, std, (out_features, in_features), requires_grad=True)
        self.bias = Tensor.normal(mean, std, (1,out_features), requires_grad=True) if bias else None
    
    def __call__(self, x: "Tensor") -> "Tensor":
        return x.linear(self.weight.T, self.bias)
    
