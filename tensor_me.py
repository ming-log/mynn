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
                 name=None
                 ):

        self.data = ensure_ndarray(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on
        self.name = name

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
            return f"tensor(array(\n{self.data}), requires_grad={self.requires_grad}, name={self.name})"
        else:
            return f"tensor(array(\n{self.data}), requires_grad={self.requires_grad})"


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

        if other.requires_grad:
            depends_on = [Dependency(self, grad_fn1), Dependency(other, grad_fn2)]
        else:
            depends_on = [Dependency(self, grad_fn1)]
        return Tensor(self.data * other.data, self.requires_grad, depends_on=depends_on)


    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        右乘法
        y = other * x
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

        if other.requires_grad:
            depends_on = [Dependency(self, grad_fn1), Dependency(other, grad_fn2)]
        else:
            depends_on = [Dependency(self, grad_fn1)]

        return Tensor(self.data * other.data, self.requires_grad, depends_on = depends_on)


    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        加法运算
        y = x + other
        """
        def grad_fn(grad):
            return grad

        other = ensure_Tensor(other)

        if other.requires_grad:
            depends_on = [Dependency(self, grad_fn), Dependency(other, grad_fn)]
        else:
            depends_on = [Dependency(self, grad_fn)]
        return Tensor(self.data + other.data, self.requires_grad, depends_on=depends_on)


    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        加法运算
        y = other + x
        """
        def grad_fn(grad):
            return grad

        other = ensure_Tensor(other)

        if other.requires_grad:
            depends_on = [Dependency(self, grad_fn), Dependency(other, grad_fn)]
        else:
            depends_on = [Dependency(self, grad_fn)]
        return Tensor(self.data + other.data, self.requires_grad, depends_on=depends_on)

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        加法运算
        y = x - other
        """
        def grad_fn1(grad):
            return grad

        def grad_fn2(grad):
            return grad * (-1)

        other = ensure_Tensor(other)

        if other.requires_grad:
            depends_on = [Dependency(self, grad_fn1), Dependency(other, grad_fn2)]
        else:
            depends_on = [Dependency(self, grad_fn1)]
        return Tensor(self.data - other.data, self.requires_grad, depends_on=depends_on)


    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        加法运算
        y = other - x
        """

        def grad_fn1(grad):
            return grad

        def grad_fn2(grad):
            return grad * (-1)

        other = ensure_Tensor(other)

        if other.requires_grad:
            depends_on = [Dependency(self, grad_fn2), Dependency(other, grad_fn1)]
        else:
            depends_on = [Dependency(self, grad_fn2)]
        return Tensor(other.data - self.data, self.requires_grad, depends_on=depends_on)


    def __pow__(self, power, modulo=None) -> 'Tensor':
        """
        power: 幂值
        """
        def grad_fn(grad):
            return grad * power * self.data ** (power - 1)
        return Tensor(self.data ** power, self.requires_grad, depends_on=[Dependency(self, grad_fn)])


    def sum(self) -> 'Tensor':
        return tensor_sum(self)

#
def tensor_sum(t: Tensor) -> Tensor:
    data = t.data.sum()
    requires_grad = t.requires_grad
    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad*np.ones_like(t.data)
        depends_on.append(Dependency(t, grad_fn))
        print('-'*100)
    return Tensor(data, requires_grad, depends_on)

# 确保输入数据为ndarray，如果不是ndarray则需要将其进行转化
def ensure_ndarray(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    return np.array(arrayable)

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
    y = -2 * a ** 2 - a
    y.backward()
    y.zero_grad()
    print(a.grad)
    print(y)
    print(y.sum())
