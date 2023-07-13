from robingrad.tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = Tensor.normal(0,1,(out_features, in_features), requires_grad=True)
        self.bias = Tensor.normal(0,1,(1,out_features), requires_grad=True) if bias else None
    
    def __call__(self, x: "Tensor") -> "Tensor":
        return x.linear(self.weight.T, self.bias)
    