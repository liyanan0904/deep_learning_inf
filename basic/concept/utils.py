#/usr/bin/env python
#-*- coding : utf-8 -*-

import numpy as np

np.seterr(all = 'ignore')

def sigmoid(x):
    return 1./(1 + np.exp(-x))

def dsigmoid(x):
    return x * (1. - x)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1. - x*x

def softmax(x):
    e = np.exp(x - np.max(x))
    if e.ndim == 1:
        return e/np.sum(e, axis = 0)
    return e/np.array([np.sum(e, axis=1)]).T

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)



