import numpy as np
from typing import Tuple, Union, List, TypeVar, Optional

DType = TypeVar("DType", bound=np.dtype)

# Trust in broadcasting rules
class Tensor:
    def __init__(self, data: Union[np.ndarray, List, int, float], dtype: DType = np.float32, requires_grad: bool = False, _children: Tuple =(), _op: str = "", _origin: str = "tensor"):
        self.data = np.asarray(data).astype(dtype)
        if requires_grad:
            self.grad = np.zeros_like(data).astype(dtype)
        else:
            self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._origin = _origin 
        self.requires_grad = requires_grad
    
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

    def sum(self) -> "Tensor":
        out = Tensor(self.data.sum(), dtype=self.data.dtype, _children=(self,), _op="sum()", _origin="sum", requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        
        return out
    
    def mean(self, axis=None, keepdim: bool = False) -> "Tensor":
        if axis is None:
            data = self.data.mean()
        else:
            data = self.data.mean(axis=axis, keepdims=keepdim)
        out = Tensor(data, dtype=self.data.dtype, _children=(self,), _op="μ", _origin="mean", requires_grad=self.requires_grad)
    
        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad / self.data.size
            else:
                axis_sum_size = self.data.shape[axis] if keepdim else 1
                self.grad += np.ones_like(self.data) * out.grad / axis_sum_size
        out._backward = _backward

        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0, self.data), dtype=self.data.dtype, _children=(self,), _op="relu()", _origin="ReLU", requires_grad=self.requires_grad)

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self) -> "Tensor":
        x = self.data
        t = (1 + np.exp(-x))**-1
        out = Tensor(t, dtype=self.data.dtype, _children=(self,), _op="sigmoid()", _origin="sigmoid", requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += t*(1-t) * out.grad 
        out._backward = _backward

        return out 
    
    def tanh(self) -> "Tensor":
        x = self.data
        t = (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
        out = Tensor(t, dtype=self.data.dtype, _children=(self,), _op="tanh()", _origin="tanh", requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out

    def exp(self) -> "Tensor":
        x = self.data
        out = Tensor(np.exp(x), dtype=self.data.dtype, _children=(self,), _op="exp()", _origin="exp", requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward
        
        return out
    
    def log(self) -> "Tensor":
        if np.any(self.data <= 0):
            raise ValueError("can't log negative or zero value")
        x = self.data
        out = Tensor(np.log(x), dtype=self.data.dtype, _children=(self,), _op="log()", _origin="log", requires_grad=self.requires_grad)

        def _backward():
            self.grad = (x**(-1)) * out.grad
        out._backward = _backward

        return out
    
    def __add__(self, other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor.broadcast(self, other, dtype=self.data.dtype, requires_grad=self.requires_grad)
        out = Tensor(self.data + other.data, dtype=self.data.dtype, _children=(self, other), _op='+', _origin="__add__", requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += out.grad
            # other.grad += out.grad
            other.grad += np.sum(out.grad, axis=0) if len(other.data.shape) > 1 else out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor.broadcast(self, other, dtype=self.data.dtype, requires_grad=self.requires_grad)
        out = Tensor(self.data * other.data, dtype=self.data.dtype, _children=(self, other), _op='*', _origin="__mul__", requires_grad=self.requires_grad)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other: Union[int, float]) -> "Tensor":
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, dtype=self.data.dtype, _children=(self,), _op=f'**{other}', _origin="__pow__", requires_grad=self.requires_grad)

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def __matmul__(self, other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype, requires_grad=self.requires_grad)
        out = Tensor(self.data @ other.data, dtype=self.data.dtype, _children=(self, other), _op='@', _origin="__matmul__", requires_grad=self.requires_grad)

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out
    
    def __rmatmul__(self, other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor":
        return self @ other
    
    def __neg__(self) -> "Tensor": # -self
        return self * Tensor.full_like(self, -1, requires_grad=self.requires_grad)

    def __radd__(self, other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # other + self
        return self + other

    def __sub__(self, other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # self - other
        return self + (-other)

    def __rsub__(self, other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # other - self
        return other + (-self)

    def __rmul__(self, other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # other * self
        return self * other

    def __truediv__(self, other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # self / other
        return self * other**-1

    def __rtruediv__(self, other: Union["Tensor", np.ndarray, List, int, float]) -> "Tensor": # other / self
        return other * self**-1
    
    def backward(self) -> None:
        if self.shape != () and self.shape != (1,) and self.shape != (1,1):
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
    def shape(self) -> Tuple[int, int]: return self.data.shape

    @property
    def dtype(self) -> str: return self.data.dtype
            
    def __repr__(self) -> str:
        if self.requires_grad:
            return f"Tensor: {self._origin}\ndata: \n{self.data}\ngrad: \n{self.grad}\ndtype: {self.data.dtype}"
        else:
            return f"Tensor: {self._origin}\ndata: \n{self.data}\ndtype: {self.data.dtype}"
    
    def __getitem__(self, val) -> "Tensor":
        out = Tensor(self.data[val], dtype=self.data.dtype, _children=(self,), _op="slice", _origin="slice", requires_grad=self.requires_grad)
        
        def _backward():
            self.grad[val] += out.grad
        out._backward = _backward

        return out

    def reshape(self, shape: Tuple[int, int]) -> "Tensor":
        out = Tensor(self.data.reshape(shape), dtype=self.data.dtype, _children=(self,), _op="reshape", _origin="reshape", requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward

        return out
    
    def transpose(self, ax1: int = 1, ax2: int = 0) -> "Tensor":
        out = Tensor(self.data.transpose(ax1, ax2), dtype=self.data.dtype, _children=(self,), _op="T", _origin="T", requires_grad=self.requires_grad)

        def _backward():
            self.grad += out.grad.transpose(ax1, ax2)
        out._backward = _backward

        return out

    @property
    def T(self) -> "Tensor": return self.transpose()  
    
    def linear(self, weight: "Tensor", bias: Optional["Tensor"] = None) -> "Tensor":
        x = self * weight if len(weight.shape) == 1 else self @ weight
        return x + bias if bias is not None else x