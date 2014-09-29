import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import scale
import time
import matplotlib.pyplot as plt
from PIL import Image
import cPickle
import copy


class Autoencoder(object):
    def __init__(self, input_tensor, n_in, n_hidden, learning_rate, pct_blackout=0.2, 
                    W=None, b_in=None, b_out=None):
        if W == None:
            # initialization of weights as suggested in theano tutorials
            W = np.asarray(np.random.uniform(
                                        low=-4 * np.sqrt(6. / (n_hidden + n_in)),
                                        high=4 * np.sqrt(6. / (n_hidden + n_in)),
                                        size=(n_in, n_hidden)), 
                                        dtype=theano.config.floatX)

        self.W = theano.shared(W, 'W')

        if b_in == None:
            self.b_in = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'b_in')
        else:
            self.b_in = theano.shared(b_in, 'b_in')

        if b_out == None:
            self.b_out = theano.shared(np.zeros(n_in, dtype=theano.config.floatX), 'b_out')
        else:
            self.b_out = theano.shared(b_out, 'b_out')

        matrixType = T.TensorType(theano.config.floatX, (False,)*2)


        self.n_in = n_in
        self.n_hidden = n_hidden
        self.inputs = input_tensor
        self.x = matrixType('x')
        self.pct_blackout = pct_blackout
        self.noise = T.shared_randomstreams.RandomStreams(1234).binomial(
                            (self.x.shape), n=1, p=1-(self.pct_blackout), 
                            dtype=theano.config.floatX)
        self.noisy = self.noise * self.x
        self.active_hidden = T.nnet.sigmoid(T.dot(self.noisy, self.W) + self.b_in)
        self.output = T.nnet.sigmoid(T.dot(self.active_hidden, self.W.T) + self.b_out)

        self.entropy = -T.sum(self.x * T.log(self.output) + 
                                (1 - self.x) * T.log(1 - self.output), axis=1)

        self.cost = T.mean(self.entropy)

        self.params = [self.W, self.b_in, self.b_out]
        self.gradients = T.grad(self.cost, self.params)

        self.learning_rate = learning_rate

        self.updates = []
        for param, grad in zip(self.params, self.gradients):
            self.updates.append((param, param - self.learning_rate * grad))

        i, batch_size = T.iscalars('i', 'batch_size')
        self.train_step = theano.function([i, batch_size], self.cost, 
                                            updates=self.updates, 
                                            givens={self.x:self.inputs[i:i+batch_size]})
                                            #, mode="DebugMode")

    def copy(self, new_learning_rate=None):
        if new_learning_rate == None:
            new_learning_rate = self.learning_rate
        return Autoencoder(self.inputs, self.n_in, self.n_hidden, new_learning_rate,
                            self.pct_blackout, self.W.get_value(), self.b_in.get_value(),
                            self.b_out.get_value())

    def save(self, f):
        params = {}
        params['W'] = self.W.get_value()
        params['b_in'] = self.b_in.get_value()
        params['b_out'] = self.b_out.get_value()
        params['n_in'] = self.n_in
        params['n_hidden'] = self.n_hidden
        params['pct_blackout'] = self.pct_blackout
        np.savez_compressed(f, **params)

    def recreate(self, img, noise=False):
        img_copy = copy.deepcopy(img)

        if "recreation_function" not in dir(self):
            matrixType = T.TensorType(theano.config.floatX, (False,)*2)
            start = matrixType('imgs')
            activation = T.nnet.sigmoid(T.dot(start, self.W) + self.b_in)
            output = T.nnet.sigmoid(T.dot(activation, self.W.T) + self.b_out)
            self.recreation_function = theano.function([start], output)

        return self.recreation_function(img)


def load(f, inputs, mask=None, original_input=None, activation=T.nnet.sigmoid):
    data = np.load(f)
    return Autoencoder(data['n_in'], data['n_hidden'], inputs, mask, data['pct_blackout'], 
                            data['W'], data['b_in'], data['b_out'], original_input, activation)

