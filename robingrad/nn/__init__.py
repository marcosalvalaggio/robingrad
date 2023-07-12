from robingrad.tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = Tensor.normal(0,1,(in_features, out_features)) # By inverting in and out num, I don't need to transpose the weight matrix.
        self.bias = Tensor.normal(0,1,(out_features,)) if bias else None
    
    def __call_(self, x: "Tensor") -> "Tensor":
        out = x @ self.weight
        if self.bias:
            out += self.bias
        return out
    