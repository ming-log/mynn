# -*- coding: utf-8 -*-
"""
@Time:2022/7/22 10:27
@Author:Ming-Log
@File:test.py
@IDE:PyCharm
"""
# from tensor import Tensor
#
# a = Tensor([1, 2, 3])
# b = Tensor([1, 2, 3])
# c = Tensor([5, 3, 1], requires_grad=True)
#
# loss = lambda a, b: ((a - b) ** 2).mean()
#
# y = loss(a, c)
# y.backward()
# print(c.grad)

# from core import initializer
#
# ini = initializer.XavierUniformInit()
# print(ini([5, 3]))
# ini2 = initializer.ZerosInit()
# print(ini2([4, 5]))
import nn
import tensor
from tensor import Tensor, relu
#
# a = Tensor([[-1, 2, 1], [3, 4, 5]], requires_grad=True, name='a')
# b = Tensor([[1, 0, 1], [1, 2, 0]], requires_grad=True, name='b')
# y = a.T@b
# y.backward()
# print(a)
# print(y)
# print(a.grad)
# print(b.grad)
#
# import torch
#
# aa = torch.tensor([[-1., 2, 1], [3, 4, 5]], requires_grad=True)
# bb = torch.tensor([[1., 0, 1], [1, 2, 0]], requires_grad=True)
# cc = aa.T@bb
# cc.backward(torch.ones_like(cc))
# print(cc)
# print(aa.grad)
# print(bb.grad)
#
c = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True)
d = Tensor(nn.eye(3), requires_grad=True)
e = c@d
e.backward()
print(e)
print(d.grad)
print(c.grad)


