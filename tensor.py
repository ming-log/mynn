# -*- coding: utf-8 -*-
"""
@Time:2022/7/22 10:52
@Author:Ming-Log
@File:tensor.py
@IDE:PyCharm
"""
from typing import List, NamedTuple, Optional, Union, Callable
# 引入类型系统，Python作为一门弱类型语言，我们需要引入类型系统，对于大系统开发而言不容易发生错误
# typing: Union[X, Y]意味着：要么是X要么是Y。定义一个联合类型，需要注意的有：
# Optional: 可选类型。等价于Union[X, None]
# Callable: Callable[[int], str]是一个函数，接受一个int参数，返回一个str

import numpy as np
import nn

Arrayable = Union[float, List, np.ndarray]



# 允许的类型
class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]  # 定义输入输出


class Tensor:
    # Tensor 数据结构是对numpy ndarray 一个包裹
    # requires_grad: 表示tensor是否参与计算梯度，在反向传播是否跟踪该tensor
    def __init__(self,
                 data:Arrayable,
                 requires_grad:bool=False,
                 depends_on: List[Dependency]=[],
                 name=None,
                 dtype=nn.float32
                 ):

        self.data = ensure_ndarray(data, dtype)
        self.requires_grad = requires_grad
        self.depends_on = depends_on
        self.name = name
        self.dtype = dtype

        self.grad: Tensor = None
        self.shape = self.data.shape
        if self.requires_grad:
            self.zero_grad()

    # 梯度置零
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))
        for tensor, grad_fn in self.depends_on:
            tensor.zero_grad()


    def backward(self, grad:'Tensor'=None) -> None:
        assert self.requires_grad, "called backward on non-requires"
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                grad = Tensor(np.ones_like(self.grad.data))
        self.grad.data += grad.data  # 梯度累加
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))


    def __repr__(self):
        if self.name:
            return f"tensor(array({self.data}), requires_grad={self.requires_grad}, name={self.name})"
        else:
            return f"tensor(array({self.data}), requires_grad={self.requires_grad})"


    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        加法运算
        y = x + other
        """
        def grad_fn(grad):
            return grad

        other = ensure_Tensor(other)

        if self.requires_grad and other.requires_grad:
            depends_on = [Dependency(self, grad_fn), Dependency(other, grad_fn)]
        elif not self.requires_grad and other.requires_grad:
            depends_on = [Dependency(other, grad_fn)]
        elif self.requires_grad and not other.requires_grad:
            depends_on = [Dependency(self, grad_fn)]
        else:
            depends_on = []
        return Tensor(self.data + other.data, self.requires_grad or other.requires_grad, depends_on=depends_on)


    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        加法运算
        y = other + x
        """
        def grad_fn(grad):
            return grad

        other = ensure_Tensor(other)


        if self.requires_grad and other.requires_grad:
            depends_on = [Dependency(self, grad_fn), Dependency(other, grad_fn)]
        elif not self.requires_grad and other.requires_grad:
            depends_on = [Dependency(other, grad_fn)]
        elif self.requires_grad and not other.requires_grad:
            depends_on = [Dependency(self, grad_fn)]
        else:
            depends_on = []
        return Tensor(other.data + self.data, self.requires_grad or other.requires_grad, depends_on=depends_on)

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        减法运算（左减）
        y = x - other
        """
        def grad_fn1(grad):
            return grad

        def grad_fn2(grad):
            return grad * (-1)

        other = ensure_Tensor(other)

        if self.requires_grad and other.requires_grad:
            depends_on = [Dependency(self, grad_fn1), Dependency(other, grad_fn2)]
        elif not self.requires_grad and other.requires_grad:
            depends_on = [Dependency(other, grad_fn2)]
        elif self.requires_grad and not other.requires_grad:
            depends_on = [Dependency(self, grad_fn1)]
        else:
            depends_on = []
        return Tensor(self.data - other.data, self.requires_grad or other.requires_grad, depends_on=depends_on)



    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        减法运算（右减）
        y = other - x
        """

        def grad_fn1(grad):
            return grad

        def grad_fn2(grad):
            return grad * (-1)

        other = ensure_Tensor(other)

        # 分情况讨论，当计算的结果不可计算梯度时，如何处理
        if self.requires_grad and other.requires_grad:
            depends_on = [Dependency(self, grad_fn2), Dependency(other, grad_fn1)]
        elif not self.requires_grad and other.requires_grad:
            depends_on = [Dependency(other, grad_fn1)]
        elif self.requires_grad and not other.requires_grad:
            depends_on = [Dependency(self, grad_fn2)]
        else:
            depends_on = []
        return Tensor(other.data - self.data, self.requires_grad or other.requires_grad, depends_on=depends_on)


    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        左乘法
        y = x * other
        """
        def grad_fn1(grad):
            """
            grad = dy/dh_{l}
            return dy/dh_{l} * dh_{l} / dh_{l-1}
            """
            return grad * other.data

        def grad_fn2(grad):
            return grad * self.data

        other = ensure_Tensor(other)

        if self.requires_grad and other.requires_grad:
            depends_on = [Dependency(self, grad_fn1), Dependency(other, grad_fn2)]
        elif not self.requires_grad and other.requires_grad:
            depends_on = [Dependency(other, grad_fn2)]
        elif self.requires_grad and not other.requires_grad:
            depends_on = [Dependency(self, grad_fn1)]
        else:
            depends_on = []
        return Tensor(self.data * other.data, other.requires_grad or self.requires_grad, depends_on=depends_on)

    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        右乘法
        y = other * x
        """
        def grad_fn1(grad):
            return grad * other.data

        def grad_fn2(grad):
            return grad * self.data

        other = ensure_Tensor(other)

        if other.requires_grad:
            depends_on = [Dependency(self, grad_fn1), Dependency(other, grad_fn2)]
        elif not self.requires_grad and other.requires_grad:
            depends_on = [Dependency(other, grad_fn2)]
        elif self.requires_grad and not other.requires_grad:
            depends_on = [Dependency(self, grad_fn1)]
        else:
            depends_on = []
        return Tensor(self.data * other.data, self.requires_grad or other.requires_grad, depends_on = depends_on)


    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        除法 (左除)
        y = x / other
        """
        def grad_fn1(grad):
            return grad * 1/other.data

        def grad_fn2(grad):
            return grad * self.data * (-1) * other.data ** (-2)

        other = ensure_Tensor(other)

        if self.requires_grad and other.requires_grad:
            depends_on = [Dependency(self, grad_fn1), Dependency(other, grad_fn2)]
        elif not self.requires_grad and other.requires_grad:
            depends_on = [Dependency(other, grad_fn2)]
        elif self.requires_grad and not other.requires_grad:
            depends_on = [Dependency(self, grad_fn1)]
        else:
            depends_on = []
        return Tensor(self.data / other.data, other.requires_grad or self.requires_grad, depends_on=depends_on)



    def __rtruediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        除法 (右除)
        y = other / x
        """

        def grad_fn1(grad):
            return grad * 1 / self.data

        def grad_fn2(grad):
            return grad * other.data * (-1) * self.data ** (-2)

        other = ensure_Tensor(other)

        if self.requires_grad and other.requires_grad:
            depends_on = [Dependency(self, grad_fn2), Dependency(other, grad_fn1)]
        elif not self.requires_grad and other.requires_grad:
            depends_on = [Dependency(other, grad_fn1)]
        elif self.requires_grad and not other.requires_grad:
            depends_on = [Dependency(self, grad_fn2)]
        else:
            depends_on = []
        return Tensor(self.data / other.data, other.requires_grad or self.requires_grad, depends_on=depends_on)


    def __pow__(self, power: Union[int, float, 'Tensor'], modulo=None) -> 'Tensor':
        """
        power: 幂值
        """
        def grad_fn(grad):
            return grad * power * self.data ** (power - 1)
        return Tensor(self.data ** power, self.requires_grad, depends_on=[Dependency(self, grad_fn)])


    def __eq__(self, other: 'Tensor') -> 'Tensor':
        """
        判断Tensor值是否等于  x == other
        """
        other = ensure_Tensor(other)
        return Tensor(self.data == other.data)

    def __lt__(self, other: 'Tensor') -> 'Tensor':
        """
        小于 x < other
        """
        other = ensure_Tensor(other)
        return Tensor(self.data < other.data)

    def __le__(self, other: 'Tensor') -> 'Tensor':
        """
        小于等于 x <= other
        """
        other = ensure_Tensor(other)
        return Tensor(self.data <= other.data)

    def __ne__(self, other: 'Tensor') -> 'Tensor':
        """
        不等于 x != other
        """
        other = ensure_Tensor(other)
        return Tensor(self.data != other.data)

    def __gt__(self, other: 'Tensor') -> 'Tensor':
        """
        大于 x > other
        """
        other = ensure_Tensor(other)
        return Tensor(self.data > other.data)

    def __ge__(self, other: 'Tensor') -> 'Tensor':
        """
        大于等于 x >= other
        """
        other = ensure_Tensor(other)
        return Tensor(self.data >= other.data)

    def sum(self) -> 'Tensor':
        return tensor_sum(self)

    def mean(self) -> 'Tensor':
        return tensor_mean(self)

# 求和函数
def tensor_sum(t: Tensor) -> Tensor:
    if t.requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad*np.ones_like(t.data)
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(t.data.sum(), t.requires_grad, depends_on=depends_on)

# 均值函数
def tensor_mean(t: Tensor) -> Tensor:
    if t.requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad*np.ones_like(t.data) / np.size(t.data)
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(t.data.mean(), t.requires_grad, depends_on=depends_on)


# 确保输入数据为ndarray，如果不是ndarray则需要将其进行转化
def ensure_ndarray(arrayable: Arrayable, dtype) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    return np.array(arrayable, dtype=dtype)

# 确保输入数据为Tensor，如果不是Tensor则需要将其进行转化
def ensure_Tensor(other: Union[Tensor, int, float, np.ndarray]) -> Tensor:
    if isinstance(other, (int, float, np.ndarray)):
        other = Tensor(other)
    return other

if __name__ == '__main__':
    a = Tensor([[5.0, 2.0, 3.0], [2, 2, 2]], requires_grad=True)
    # b = 20
    # def loss(a, b):
    #     return a**3 + b**2 + 2*a
    # y = loss(a, b)
    # y = -2 * a ** 2 - a
    # y.backward()
    # y.zero_grad()
    # print(a.grad)
    # print(y)
    # print(y.sum())

    a = Tensor([1, 2, 3])
    b = Tensor([1, 2, 3])
    c = Tensor([5, 3, 1], requires_grad=True)

    y = a * c
    y.backward()

    print(c.grad)
