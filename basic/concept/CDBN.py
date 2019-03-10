#!/usr/bin/env python
#-*- coding : utf-8 -*-

import sys
import numpy as np
from HiddenLayer import HiddenLayer
from LR import LR
from RBM import RBM
from CRBM import CRBM
from DBN import DBN
from utils import *


class CDBN(DBN):
    def __init__(self, input=None, label=None, n_ins=2, hidden_layer_sizes=[3,3], n_outs=2, rng=None):

        self.x = input
        self.y = label
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)

        if rng is None:
            rng = np.random.RandomState(1234)

        assert self.n_layers > 0

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()

            sigmoid_layer = HiddenLayer(input = layer_input, n_in=input_size, n_out=hidden_layer_sizes[i], rng=rng, activation=sigmoid)

            self.sigmoid_layers.append(sigmoid_layer)

            # construct rbm_layer

            if  i == 0:
                rbm_layer = CRBM(input=layer_input, n_visible=input_size, n_hidden=hidden_layer_sizes[i], W=sigmoid_layer.W, hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layer_sizes[i],
                                W=sigmoid_layer.W,     # W, b are shared
                                hbias=sigmoid_layer.b)


            self.rbm_layers.append(rbm_layer)

        self.log_layer = LR(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        self.finetune_cost =self.log_layer.negative_log_likelihood()



def test_cdbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1, finetune_lr=0.1, finetune_epochs=200):
    x = np.array([[0.4, 0.5, 0.5, 0.,  0.,  0.],
                     [0.5, 0.3,  0.5, 0.,  0.,  0.],
                     [0.4, 0.5, 0.5, 0.,  0.,  0.],
                     [0.,  0.,  0.5, 0.3, 0.5, 0.],
                     [0.,  0.,  0.5, 0.4, 0.5, 0.],
                     [0.,  0.,  0.5, 0.5, 0.5, 0.]])

    y= np.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]])
    rng = np.random.RandomState(123)

    dbn = CDBN(input=x, label=y, n_ins=6, hidden_layer_sizes=[5, 5], n_outs=2, rng=rng)

    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)

    x = np.array([[0.5, 0.5, 0., 0., 0., 0.],
                     [0., 0., 0., 0.5, 0.5, 0.],
                     [0.5, 0.5, 0.5, 0.5, 0.5, 0.]])

    print dbn.predict(x)


if __name__ == "__main__":
    test_cdbn()
