# -*- coding: utf-8 -*-
"""
@Time:2022/7/26 15:36
@Author:Ming-Log
@File:loss.py
@IDE:PyCharm
"""
import nn
# loss.py
class BaseLoss(object):
    def loss(self, predicted, actual):
        raise NotImplementedError

    def grad(self, predicted, actual):
        raise NotImplementedError


class CrossEntropyLoss(BaseLoss):
    def loss(self, predicted, actual):
        m = predicted.shape[0]
        exps = nn.exp(predicted - nn.max(predicted, axis=1, keepdims=True))
        p = exps / nn.sum(exps, axis=1, keepdims=True)
        nll = -nn.log(nn.sum(p * actual, axis=1))
        return nn.sum(nll) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        grad = nn.copy(predicted)
        grad -= actual
        return grad / m
