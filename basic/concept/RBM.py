#!/usr/bin/env python
#-*- coding : utf-8 -*-

import sys
import numpy as np
from utils import *

class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, W=None, hbias=None, vbias=None, rng=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if rng is None:
            rng = np.random.RandomState(1234)

        if W is None:
            a = 1./n_visible
            initial_W = np.array(rng.uniform(low = -a, high=a, size=(n_visible, n_hidden)))
            W = initial_W

        if hbias is None:
            hbias = np.zeros(n_hidden)

        if vbias is None:
            vbias = np.zeros(n_visible)

        self.rng = rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

    def contrastive_divergence(self, lr=0.1, k=1, input=None):
        if input is not None:
            self.input = input

        '''CD-K'''
        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        chain_start = ph_sample

        for step in xrange(k):
            if step == 0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        self.W += lr * (np.dot(self.input.T, ph_mean) - np.dot(nv_samples.T, nh_means))

        self.vbias += lr * np.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * np.mean(ph_mean - nh_means, axis=0)



    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)

        return [h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.rng.binomial(size=v1_mean.shape, n=1, p=v1_mean)
        return [v1_mean, v1_sample]


    def propup(self, v):
        pre_sigmoid_activation = np.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = np.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)

    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample, h1_mean, h1_sample]

    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = np.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W.T) + self.vbias

        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy =-np.mean(np.sum(self.input * np.log(sigmoid_activation_v) + (1-self.input)*np.log(1-sigmoid_activation_v), axis=1))

        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(np.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(np.dot(h, self.W.T) + self.vbias)

        return reconstructed_v


def test_rbm(learning_rate=0.1, k=1, training_epochs=1000):
    data = np.array([[1,1,1,0,0,0],
                        [1,0,1,0,0,0],
                        [1,1,1,0,0,0],
                        [0,0,1,1,1,0],
                        [0,0,1,1,0,0],
                        [0,0,1,1,1,0]])

    rng = np.random.RandomState(123)

    rbm = RBM(input=data, n_visible=6, n_hidden=2, rng=rng)

    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)

    v = np.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0]])

    print rbm.reconstruct(v)


if __name__ == "__main__":

    test_rbm()








