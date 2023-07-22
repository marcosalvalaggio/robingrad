## Robingrad

<h1 align="center">
<img src="logo.png" width="150">
</h1><br>


[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![stability-wip](https://img.shields.io/badge/stability-wip-lightgrey.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#work-in-progress)

Something between [Tinygrad](https://github.com/tinygrad/tinygrad) and [Micrograd](https://github.com/karpathy/micrograd).


## Installation

To install the current release,

```console
pip install robingrad==0.0.6
```

From source

```console
git clone https://github.com/marcosalvalaggio/robingrad.git
cd robingrad
./build.sh
```

## Examples

* In the [examples](examples/) folder, you can find examples of models trained using the **Robingrad** library.
* A declaration example of an MLP net using **Robingrad**:


```python 
from robingrad import Tensor, draw_dot
import robingrad.nn as nn

class RobinNet:
    def __init__(self):
        self.l1 = nn.Linear(5,16)
        self.l2 = nn.Linear(16,1)
    def __call__(self, x):
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        return x
        
model = RobinNet()
res = model(X_train[0].reshape((1,5)))
draw_dot(res)
```
