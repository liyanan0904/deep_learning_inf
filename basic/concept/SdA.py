# -*- coding:utf-8 -*-

import sys
import numpy as np
from HiddenLayer import HiddenLayer
from LR import LR
from dA import dA
from utils import *


class SdA(object):

    def __init__(self, input=None, label=None, n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2, rng=None):

        self.x = input
        self.y = label

        self.sigmoid_layers = []
        self.dA_layers = []

        self.n_layers = len(hidden_layer_sizes)

        if rng is None:
            rng = np.random.RandomState(1234)

        assert self.n_layers > 0


        # construct multi-layer
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()

            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input, n_in=input_size, n_out=hidden_layer_sizes[i], rng=rng, activation=sigmoid)

            self.sigmoid_layers.append(sigmoid_layer)

            dA_layer = dA(input=layer_input, n_visible=input_size, n_hidden=hidden_layer_sizes[i], W=sigmoid_layer.W, hbias=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

        self.log_layer = LR(input=self.sigmoid_layers[-1].sample_h_given_v(), label=self.y, n_in=hidden_layer_sizes[-1], n_out=n_outs)

        self.finetune_cost = self.log_layer.negative_log_likelihood()


    def pretrain(self, lr=0.1, corruption_level=0.3, epochs=100):
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i - 1].sample_h_given_v(layer_input)

            da = self.dA_layers[i]

            for epoch in xrange(epochs):
                da.train(lr=lr, corruption_level=corruption_level, input=layer_input)

    def finetune(self, lr=0.1, epochs=100):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()
        # train log_layer
        epoch = 0

        while epoch < epochs:
            self.log_layer.train(lr=lr, input=layer_input)

            lr *= 0.95
            epoch += 1

    def predict(self, x):
        layer_input = x

        for i in xrange(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)

        return self.log_layer.predict(layer_input)


def test_SdA(pretrain_lr=0.1, pretraining_epochs=1000, corruption_level=0.3, finetune_lr=0.1, finetune_epochs=200):
    x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    y = np.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1],
                     [0, 1],
                     [0, 1]])
    rng = np.random.RandomState(123)

    # construct SdA
    sda = SdA(input=x, label=y, n_ins=20, hidden_layer_sizes=[15, 15], n_outs=2, rng=rng)


    # pre-training
    sda.pretrain(lr=pretrain_lr, corruption_level=corruption_level, epochs=pretraining_epochs)

    # fine-tuning
    sda.finetune(lr=pretrain_lr, epochs=finetune_epochs)

    # test
    x = np.array([[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]])

    print sda.predict(x)


if __name__ == "__main__":
    test_SdA()



