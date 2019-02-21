#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import numpy as np


from utils import *

class LR(object):

    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label

        self.W = np.zeros((n_in, n_out))

        self.b = np.zeros(n_out)

    def train(self, lr=0.1, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input

        p_y_given_x = self.output(self.x)
        d_y = self.y - p_y_given_x

        self.W += lr * np.dot(self.x.T, d_y) - lr * L2_reg * self.W

        self.b += lr * np.mean(d_y, axis = 0)

        self.d_y = d_y

    def output(self, x):
        return softmax(np.dot(x, self.W) + self.b)

    def predict(self, x):
        return self.output(x)

    def negative_log_likelihood(self):
        sigmoid_activation = softmax(np.dot(self.x, self.W) + self.b)

        cross_entropy = -np.mean(np.sum(self.y * np.log(sigmoid_activation) + (1 - self.y) * np.log(1 - sigmoid_activation), axis=1))


        return cross_entropy




def predict_lr(learning_rate=0.1, n_epochs=500):

    rng = np.random.RandomState(123)

    d = 2
    N = 10

    x1 = rng.randn(N, d) + np.array([0, 0])
    x2 = rng.randn(N, d) + np.array([20, 10])
    y1 = [[1, 0] for i in xrange(N)]
    y2 = [[0, 1] for i in xrange(N)]

    x = np.r_[x1.astype(int), x2.astype(int)]
    y = np.r_[y1, y2]

    classifier = LR(input=x, label=y, n_in=d, n_out=2)


    for epoch in xrange(n_epochs):

        classifier.train(lr=learning_rate)

        learning_rate *= 0.995

    result = classifier.predict(x)

    for i in xrange(N):
        print result[i]

    for i in xrange(N):
        print result[N + i]

if __name__ == "__main__":
    predict_lr()
