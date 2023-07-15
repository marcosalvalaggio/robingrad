from .tensor import Tensor
from typing import List, Dict
from collections import OrderedDict

# Tinygrad is the best framework ever.
def get_state_dict(obj, prefix:str='', tensor_type=Tensor) -> Dict[str, Tensor]:
  """
  Recursively converts an object and its nested elements into a state dictionary.

  Args:
      obj: The object to convert into a state dictionary.
      prefix (str): Optional. The prefix to prepend to each key in the state dictionary.
      tensor_type: Optional. The tensor type to use in the state dictionary.

  Returns:
      Dict[str, Tensor]: The state dictionary representing the object and its nested elements.
  """
  if isinstance(obj, tensor_type): return {prefix.strip('.'):obj}
  if hasattr(obj, '_asdict'): return get_state_dict(obj._asdict(), prefix, tensor_type)  # namedtuple
  if isinstance(obj, OrderedDict): return get_state_dict(dict(obj), prefix, tensor_type)
  if hasattr(obj, '__dict__'): return get_state_dict(obj.__dict__, prefix, tensor_type)
  state_dict = {}
  if isinstance(obj, (list, tuple)):
    for i,x in enumerate(obj): state_dict.update(get_state_dict(x, f"{prefix}{str(i)}.", tensor_type))
  elif isinstance(obj, dict):
    for k,v in obj.items(): state_dict.update(get_state_dict(v, f"{prefix}{str(k)}.", tensor_type))
  return state_dict

def get_parameters(obj) -> List[Tensor]: return list(get_state_dict(obj).values())