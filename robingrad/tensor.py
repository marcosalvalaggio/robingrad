import numpy as np
from typing import Tuple, Union, List, TypeVar

DType = TypeVar("DType", bound=np.dtype)

# Trust in broadcasting rules
class Tensor:
    def __init__(self, data: Union[np.ndarray, List, int, float], dtype: DType = np.float32, _children: Tuple =(), _op: str = "", _origin: str = "tensor"):
        self.data = np.asarray(data).astype(dtype)
        self.grad = np.zeros_like(data).astype(dtype)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._origin = _origin 
    
    @staticmethod
    def zeros(shape: Tuple, **kwargs) -> "Tensor": return Tensor(np.zeros(shape), _origin="zeros", **kwargs)

    @staticmethod
    def zeros_like(tensor: "Tensor", **kwargs) -> "Tensor": return Tensor(np.zeros_like(tensor.data), _origin="zeros_like", **kwargs)

    @staticmethod
    def eye(size: int, **kwargs) -> "Tensor": return Tensor(np.eye(size), _origin="eye", **kwargs)

    @staticmethod
    def full(shape: Tuple, fill_value: Union[float, int], **kwargs) -> "Tensor": return Tensor(np.full(shape,fill_value), _origin="full", **kwargs)

    @staticmethod
    def full_like(tensor: "Tensor", fill_value: Union[float, int], **kwargs) -> "Tensor": return Tensor(np.full_like(tensor.data, fill_value), _origin="full like", **kwargs)

    @staticmethod
    def ones(shape: Tuple, **kwargs) -> "Tensor": return Tensor(np.ones(shape), _origin="ones", **kwargs)

    @staticmethod
    def ones_like(tensor: "Tensor", **kwargs) -> "Tensor": return Tensor(np.ones_like(tensor.data), _origin="ones like", **kwargs)
    
    @staticmethod
    def broadcast(reference: "Tensor", target: "Tensor", **kwargs) -> "Tensor": 
        if np.broadcast(reference, target).shape == np.broadcast(reference, reference).shape:
            return Tensor(np.broadcast_to(target, reference.shape), _origin="broadcasted", **kwargs)
        else:
            raise ValueError("Tensors are not broadcastable.")
        
    @staticmethod
    def normal(mean: Union[int, float], std: Union[int, float], shape: Tuple, **kwargs) -> "Tensor": return Tensor(np.random.normal(mean, std, shape), _origin="normal", **kwargs)
        
    @staticmethod
    def uniform(low: Union[int, float], high: Union[int, float], shape: Tuple, **kwargs) -> "Tensor": return Tensor(np.random.uniform(low, high, shape), _origin="uniform", **kwargs)    

    def sum(self: "Tensor") -> "Tensor":
        out = Tensor(self.data.sum(), dtype=self.data.dtype, _children=(self,), _op="sum()", _origin="sum")
        
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        
        return out
    
    def __add__(self: "Tensor", other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor.broadcast(self, other, dtype=self.data.dtype)
        out = Tensor(self.data + other.data, dtype=self.data.dtype, _children=(self, other), _op='+', _origin="__add__", )
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self: "Tensor", other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor.broadcast(self, other, dtype=self.data.dtype)
        out = Tensor(self.data * other.data, dtype=self.data.dtype, _children=(self, other), _op='*', _origin="__mul__")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self: "Tensor", other: Union[int, float]) -> "Tensor":
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, dtype=self.data.dtype, _children=(self,), _op=f'**{other}', _origin="__pow__")

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def __neg__(self: "Tensor") -> "Tensor": # -self
        return self * Tensor.full_like(self, -1)

    def __radd__(self: "Tensor", other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # other + self
        return self + other

    def __sub__(self: "Tensor", other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # self - other
        return self + (-other)

    def __rsub__(self: "Tensor", other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # other - self
        return other + (-self)

    def __rmul__(self: "Tensor", other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # other * self
        return self * other

    def __truediv__(self: "Tensor", other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # self / other
        return self * other**-1

    def __rtruediv__(self: "Tensor", other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # other / self
        return other * self**-1
    
    def backward(self: "Tensor") -> None:
        if self.shape != () and self.shape != (1,):
            raise ValueError("Backward can only be called on a scalar tensor.")
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
            
    @property
    def shape(self: "Tensor") -> Tuple[int, int]: return self.data.shape
            
    def __repr__(self: "Tensor") -> str:
        return f"Tensor: {self._origin}\ndata: \n{self.data}\ngrad: \n{self.grad}\ndtype: {self.data.dtype}\n"