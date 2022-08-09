# -*- coding: utf-8 -*-
"""
@Time:2022/8/9 17:43
@Author:Ming-Log
@File:functional.py
@IDE:PyCharm
"""
from tensor import Tensor
import numpy as np

def normal(loc=0.0, scale=1.0, size=None, required_grad=False) -> 'Tensor':
    return Tensor(np.random.normal(loc=loc, scale=scale, size=size), requires_grad=required_grad)
