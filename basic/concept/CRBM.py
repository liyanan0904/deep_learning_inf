#!/usr/bin/env python
#-*- coding : utf-8 -*-

import sys
import numpy as np
from RBM import RBM
from utils import *


class CRBM(RBM):
    def propdown(self, h):
        pre_activation = np.dot(h, self.W.T) + self.vbias
        return pre_activation

    def sample_v_given_h(self, h0_sample):
        a_h = self.propdown(h0_sample)
        en = np.exp(-a_h)
        ep = np.exp(a_h)

        v1_mean = 1/(1 - en) - 1/a_h
        U = np.array(self.rng.uniform(low=0, high=1, size=v1_mean.shape))

        v1_sample = np.log((1 - U*(1-ep)))/ a_h
        return [v1_mean, v1_sample]

def test_crbm(learning_rate=0.1, k=1, training_epochs=1000):
    data = np.array([[0.4, 0.5, 0.5, 0.,  0.,  0.],
                        [0.5, 0.3,  0.5, 0.,  0.,  0.],
                        [0.4, 0.5, 0.5, 0.,  0.,  0.],
                        [0.,  0.,  0.5, 0.3, 0.5, 0.],
                        [0.,  0.,  0.5, 0.4, 0.5, 0.],
                        [0.,  0.,  0.5, 0.5, 0.5, 0.]])

    rng = np.random.RandomState(123)

    # construct CRBM
    rbm = CRBM(input=data, n_visible=6, n_hidden=5, rng=rng)

    # train
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)

    # test

    v = np.array([[0.5, 0.5, 0., 0., 0., 0.],
                     [0., 0., 0., 0.5, 0.5, 0.]])

    print rbm.reconstruct(v)

if __name__ == "__main__":
    test_crbm()
