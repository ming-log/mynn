# -*- coding: utf-8 -*-
"""
@Time:2022/7/26 15:48
@Author:Ming-Log
@File:initializer.py
@IDE:PyCharm
"""
from nn import random, zeros
from tensor import Tensor

class Init:
    def __init__(self):
        self.outputs = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class XavierUniformInit(Init):
    def __init__(self):
        super().__init__()

    def __call__(self, size=[]):
        self.outputs = random.uniform(size)
        return Tensor(self.outputs)

class ZerosInit(Init):
    def __init__(self):
        super().__init__()

    def __call__(self, size=[]):
        self.outputs = zeros(size)
        return Tensor(self.outputs)
