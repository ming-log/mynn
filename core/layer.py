# -*- coding: utf-8 -*-
"""
@Time:2022/7/26 15:36
@Author:Ming-Log
@File:layer.py
@IDE:PyCharm
"""
import numpy as np
from initializer import XavierUniformInit, ZerosInit

class Layer:
    def __init__(self, name=None):
        self.name = name
        self.params, self.grads = None, None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self,
                 num_in,
                 num_out,
                 w_init=XavierUniformInit(),
                 b_init=ZerosInit()):
        super().__init__('Linear')

        self.params = {
            "w": w_init([num_in, num_out]),
            "b": b_init([1, num_out])
        }
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads["w"] = self.inputs.T @ grad
        self.grads["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T


# layer.py
class Activation(Layer):
    """Base activation layer"""

    def __init__(self, name):
        super().__init__(name)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative_func(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative_func(self, x):
        raise NotImplementedError


class ReLU(Activation):
    """ReLU activation function"""

    def __init__(self):
        super().__init__("ReLU")

    def func(self, x):
        return np.maximum(x, 0.0)

    def derivative_func(self, x):
        return x > 0.0