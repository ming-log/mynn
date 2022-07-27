# -*- coding: utf-8 -*-
"""
@Time:2022/7/22 10:27
@Author:Ming-Log
@File:test.py
@IDE:PyCharm
"""
# from tensor import Tensor

# a = Tensor([1, 2, 3])
# b = Tensor([1, 2, 3])
# c = Tensor([5, 3, 1], requires_grad=True)
#
# loss = lambda a, b: ((a - b) ** 2).mean()
#
# y = loss(a, c)
# y.backward()

from core import initializer

ini = initializer.XavierUniformInit()
print(ini([5, 3]))
ini2 = initializer.ZerosInit()
print(ini2([4, 5]))
