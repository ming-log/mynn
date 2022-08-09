# -*- coding: utf-8 -*-
"""
@Time:2022/7/22 10:27
@Author:Ming-Log
@File:test.py
@IDE:PyCharm
"""
import torch

from tensor import Tensor
from core.functional import normal
import matplotlib.pyplot as plt

# 使用tensor类实现线性回归
## 使用线性模型y=x1w1+x2w2+b+eplsion生成数据
w_true = Tensor([[2], [-3.4]])
b_true = Tensor([[4.2]])
X = normal(0, 1, (1000, len(w_true)))
y = X@w_true + b_true
y += normal(0, 0.01, y.shape)

# 绘制散点图
plt.scatter(X[:, 1].data, y.data, s=5)
plt.show()

# 初始化模型参数
w = normal(0, 0.01, size=(2, 1), required_grad=True)
w.name='w'
b = normal(size=(1, ), required_grad=True)
b.name='b'

# 定义模型
def linear(X, w, b):
    return X@w + b


# 定义损失函数
def loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# y = linear(X, w, b)
# y.backward()
# print(w.grad)
y_hat = linear(X, w, b)
l = loss(y_hat, y)
l.backward()

XX = torch.tensor(X.data)
yy = torch.tensor(y.data)
ww = torch.tensor(w.data, requires_grad=True)
bb = torch.tensor(b.data, requires_grad=True)

yy_hat = linear(XX, ww, bb)
l = loss(yy_hat, yy)
l.mean().backward()