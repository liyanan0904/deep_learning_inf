#!/usr/bin/env python
#-*- coding : utf-8 -*-

import sys
import numpy as np

from utils import *


class dA(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, W=None, hbias=None, vbias=None, rng=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if rng is None:
            rng = np.random.RandomState(1234)

        if W is None:
            a = 1./n_visible
            W = np.array(rng.uniform(low=-a, high=a,size=(n_visible, n_hidden)))

        if hbias is None:
            hbias = np.zeros(n_hidden)

        if vbias is None:
            vbias = np.zeros(n_visible)

        self.rng = rng
        self.x = input
        self.W = W
        self.W_prime = self.W.T
        self.hbias = hbias
        self.vbias = vbias

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1

        return self.rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

    #Encode
    def get_hidden_values(self, input):
        return sigmoid(np.dot(input, self.W) + self.hbias)

    #Decode
    def get_reconstructed_input(self, hidden):
        return sigmoid(np.dot(hidden, self.W_prime) + self.vbias)

    def train(self, lr=0.1, corruption_level=0.3, input=None):
        if input is not None:
            self.x = input

        x = self.x
        tilde_x = self.get_corrupted_input(x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L_h2 = x - z

        L_h1 = np.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1

        L_W = np.dot(tilde_x.T, L_h1) + np.dot(L_h2.T, y)

        self.W += lr * L_W

        self.hbias += lr * np.mean(L_hbias, axis = 0)
        self.vbias += lr * np.mean(L_vbias, axis = 0)

    def negative_log_likelihood(self, corruption_level=0.3):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)

        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        cross_entropy = -np.mean(np.sum(self.x * np.log(z) + (1 - self.x) * np.log(1 - z), axis = 1))

        return cross_entropy

    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)

        return z


def test_dA(learning_rate=0.1, corruption_level=0.3, training_epochs=50):
    data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    rng = np.random.RandomState(123)

    da = dA(input=data, n_visible=20, n_hidden=5, rng=rng)

    #train

    for epoch in xrange(training_epochs):
        da.train(lr=learning_rate, corruption_level=corruption_level)

    x = np.array([[1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]])

    print da.reconstruct(x)


if __name__ == "__main__":
    test_dA()

