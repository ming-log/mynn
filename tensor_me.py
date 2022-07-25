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

# 确保输入数据为ndarray，如果不是ndarray则需要将其进行转化
def ensure_ndarray(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    return np.array(arrayable)

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
        self.name=name

        self.grad: Tensor = None
        self.shape = self.data.shape
        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad:'Tensor'=None) -> None:
        # print(self)
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

        if isinstance(other, (float, int)):
            new = Tensor(
                self.data * other,
                self.requires_grad,
                depends_on = [Dependency(self, grad_fn2)]
            )
        else:
            new = Tensor(
                self.data * other.data, # 正向计算值
                self.requires_grad,
                depends_on = [Dependency(self, grad_fn1), Dependency(other, grad_fn2)]
            )
        return new

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

        if isinstance(other, (float, int)):
            new = Tensor(
                self.data * other,
                self.requires_grad,
                depends_on = [Dependency(self, grad_fn2)]
            )
        else:
            new = Tensor(
                self.data * other.data,  # 正向计算值
                self.requires_grad,
                depends_on=[Dependency(self, grad_fn1), Dependency(other, grad_fn2)]
            )
        return new

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        加法运算
        y = x + other
        """
        def grad_fn(grad):
            return grad

        new = Tensor(
            self.data + other.data,  # 正向计算值
            self.requires_grad,
            depends_on=[Dependency(self, grad_fn), Dependency(self, grad_fn)]
        )
        return new

    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        加法运算
        y = other + x
        """
        def grad_fn(grad):
            return grad
        new = Tensor(
            self.data + other.data,  # 正向计算值
            self.requires_grad,
            depends_on=[Dependency(self, grad_fn), Dependency(self, grad_fn)]
        )
        return new

    def __pow__(self, power, modulo=None) -> 'Tensor':
        """
        power: 幂值
        """
        def grad_fn(grad):
            return grad * power * self.data ** (power - 1)
        new = Tensor(
            self.data ** power,  # 正向计算值
            self.requires_grad,
            depends_on=[Dependency(self, grad_fn)])
        return new


    def sum(self) -> 'Tensor':
        return tensor_sum(self)


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

if __name__ == '__main__':
    # t = [1, 2, 3]
    # t = Tensor(t, requires_grad=True)
    # sum1 = tensor_sum(t)
    # sum1.backward()
    # print(t.grad)
    a = Tensor([5.0, 2.0, 3.0], requires_grad=True)
    m = 2
    b = a ** 2
    c = b ** 2
    d = b ** 2
    y = m * c + d
    y.backward()
    print(c.grad)
    # print(a.sum)
