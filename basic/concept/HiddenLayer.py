#-*- coding : utf-8 -*-

import sys
import numpy as np
from utils import *


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None, rng=None, activation=tanh):
        if rng is None:
            rng = np.random.RandomState(1234)

        if W is None:
            a = 1. / n_in
            W = np.array(rng.uniform(low=-a, high=a, size=(n_in, n_out)))

        if b is None:
            b = np.zeros(n_out)

        self.rng = rng
        self.x = input

        self.W = W
        self.b = b

        if activation == tanh:
            self.dactivation = dtanh
        elif activation == sigmoid:
            self.dactivation = dsigmoid
        elif activation == ReLU:
            self.dactivation = dReLU
        else:
            raise ValueError('activation function not supported.')
        self.activation = activation

    def output(self, input=None):
        if input is not None:
            self.x = input

        linear_output = np.dot(self.x, self.W) + self.b
        return self.activation(linear_output)


    def forward(self, input=None):
        return self.output(input=input)


    def backward(self, prev_layer, lr=0.1, input=None, dropout=False, mask=None):
        if input is not None:
            self.x = input

        d_y = self.dactivation(prev_layer.x) * np.dot(prev_layer.d_y, prev_layer.W.T)

        if dropout == True:
            d_y *= mask

        self.W += lr * np.dot(self.x.T, d_y)
        self.b += lr * np.mean(d_y, axis=0)
        self.d_y = d_y

    def dropout(self, input, p, rng=None):
        if rng is None:
            rng = np.random.RandomState(123)

        mask = rng.binomial(size=input.shape, n=1, p=1-p)

        return mask

    def sample_h_given_v(self, input=None):
        if input is not None:
            self.x = input

        v_mean = self.output()

        h_sample = self.rng.binomial(size=v_mean.shape, n=1, p=v_mean)

        return h_sample