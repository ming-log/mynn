# -*- coding: utf-8 -*-
"""
@Time:2022/7/22 9:27
@Author:Ming-Log
@File:计算图.py
@IDE:PyCharm
"""
import numpy as np


class Tensor():
    """
    基础量，是计算图中的节点
    以数值计算为例，矩阵运算和卷积计算同理
    """
    def __init__(self, data, depend=[], name="None"):
        """
        data: 节点值
        depend: 当前节点的输入节点
        name: 节点名字
        """
        self.data = data
        self.depend = depend
        self.name = name
        self.grad = 0  # 初始梯度为0

    def __mul__(self, other):
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

        new = Tensor(
            self.data * other.data, # 正向计算值
            depend = [(self, grad_fn1), (other, grad_fn2)])
        return new

    def __rmul__(self, other):
        """
        右乘法
        y = other * x
        """

        def grad_fn1(grad):
            return grad * other.data

        def grad_fn2(grad):
            return grad * self.data
        new = Tensor(
            self.data * other.data,  # 正向计算值
            depend=[(self, grad_fn1), (other, grad_fn2)])
        return new

    def __add__(self, other):
        """
        加法运算
        y = x + other
        """
        def grad_fn(grad):
            return grad
        new = Tensor(
            self.data + other.data,  # 正向计算值
            depend=[(self, grad_fn), (other, grad_fn)])
        return new

    def __radd__(self, other):
        """
        加法运算
        y = other + x
        """
        def grad_fn(grad):
            return grad
        new = Tensor(
            self.data + other.data,  # 正向计算值
            depend=[(self, grad_fn), (other, grad_fn)])
        return new

    def __pow__(self, power, modulo=None):
        """
        power: 幂值
        """
        def grad_fn(grad):
            return grad * power * self.data ** (power - 1)
        new = Tensor(
            self.data ** power,  # 正向计算值
            depend=[(self, grad_fn)])
        return new

    def backward(self, grad=None):
        if grad == None:
            self.grad = 1
            grad = 1
        else:
            self.grad += grad
        # 递归计算每一个节点
        for tensor, grad_fn in self.depend:
            bw = grad_fn(grad)
            tensor.backward(bw)



    def zero_grad(self):
        self.grad = 0
        for tensor, grad_fn in self.depend:
            tensor.zero_grad()

x = Tensor(2)  # 定义初始输入节点
x2 = x * x  # 乘法
g = x2 * x2
h = x2 * x2
y = g + h
x2.backward()

print(x2.grad)
print(x.grad)
