# -*- coding: utf-8 -*-
"""
@Time:2022/7/22 10:27
@Author:Ming-Log
@File:test.py
@IDE:PyCharm
"""
import torch

a = torch.tensor([5.0, 2.0, 3.0], requires_grad=True)
b = a * a
c = b * b
d = b * b
y = 2*c + d
y.backward(torch.ones_like(c))
print(a.grad)
